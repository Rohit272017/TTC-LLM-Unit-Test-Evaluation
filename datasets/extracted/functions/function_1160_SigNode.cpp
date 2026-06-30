#include "tensorflow/core/grappler/graph_analyzer/sig_node.h"
#include <algorithm>
#include "absl/strings/str_format.h"
namespace tensorflow {
namespace grappler {
namespace graph_analyzer {
static constexpr bool debug = false;
SigNode::SigNode(const NodeDef* node) : node_(node) {}
void SigNode::CopyLinks(const GenNode& from, const TranslationMap& map) {
  hash_to_link_.clear();
  hashed_peers_.clear();
  std::map<LinkTag, Link> link_map;
  CopyLinksPass1(from, map, &link_map);
  CopyLinksPass2(&link_map);
}
void SigNode::CopyLinksPass1(const GenNode& from, const TranslationMap& map,
                             std::map<LinkTag, Link>* link_map) {
  LinkTag::Hasher link_hasher;
  for (const auto& entry : from.links()) {
    for (const auto& target : entry.second) {
      auto nodeit = map.find(target.node);
      if (nodeit == map.end()) {
        continue;
      }
      LinkTag tag(entry.first, target.port);
      size_t hval = link_hasher(tag);
      Link& map_entry = (*link_map)[tag];
      if (map_entry.peers.empty()) {
        map_entry.tag = tag;
        map_entry.unique_hash = hval;
      }
      map_entry.peers.push_back(nodeit->second);
    }
  }
}
void SigNode::CopyLinksPass2(std::map<LinkTag, Link>* link_map) {
  for (auto& entry : *link_map) {
    Link* hl_entry_ptr = &hash_to_link_[entry.second.unique_hash];
    while (!hl_entry_ptr->peers.empty()) {
      CombineHash(1, &entry.second.unique_hash);
      hl_entry_ptr = &hash_to_link_[entry.second.unique_hash];
    }
    for (const auto& peer : entry.second.peers) {
      hashed_peers_.emplace_back(HashedPeer(entry.second.unique_hash, peer));
    }
    hl_entry_ptr->tag = entry.second.tag;
    hl_entry_ptr->unique_hash = entry.second.unique_hash;
    hl_entry_ptr->peers.swap(entry.second.peers);
  }
}
void SigNode::ComputeTopoHash0() {
  topo_hash_.clear();
  last_hashed_nodes_ = next_hashed_nodes_ = node_mask_;
  size_t hval = std::hash<string>()(opcode());
  for (const auto& entry : hashed_peers_) {
    CombineHash(entry.link_hash, &hval);
  }
  topo_hash_.push_back(hval);
}
void SigNode::ComputeTopoHash(int distance) {
  next_hashed_nodes_ = last_hashed_nodes_;
  if (debug) {
    LOG(INFO) << "DEBUG    node " << name() << " mask=" << std::hex
              << next_hashed_nodes_;
  }
  if (hash_is_final_) {
    return;
  }
  const int64_t topo_hash_size = topo_hash_.size();
  CHECK(topo_hash_size == distance);
  int prev = distance - 1;
  size_t hval = topo_hash_[0];
  if (!hashed_peers_.empty()) {
    size_t last_link_hash = hashed_peers_[0].link_hash;
    size_t comm_hash = 0;
    for (const auto& entry : hashed_peers_) {
      if (entry.link_hash != last_link_hash) {
        CombineHash(last_link_hash, &hval);
        CombineHash(comm_hash, &hval);
        comm_hash = 0;
        last_link_hash = entry.link_hash;
      }
      CombineHashCommutative(entry.peer->GetTopoHash(prev), &comm_hash);
      next_hashed_nodes_ |= entry.peer->last_hashed_nodes_;
      if (debug) {
        LOG(INFO) << "DEBUG    node " << name() << " += " << entry.peer->name()
                  << " mask=" << std::hex << next_hashed_nodes_;
      }
    }
    CombineHash(last_link_hash, &hval);
    CombineHash(comm_hash, &hval);
  }
  topo_hash_.push_back(hval);
}
size_t SigNode::GetTopoHash(int distance) const {
  CHECK(!topo_hash_.empty());
  const int64_t topo_hash_size = topo_hash_.size();
  if (distance >= topo_hash_size) {
    CHECK(hash_is_final_);
    return topo_hash_.back();
  } else {
    return topo_hash_[distance];
  }
}
bool SigNode::operator==(const SigNode& other) const {
  if (opcode() != other.opcode()) {
    return false;
  }
  if (unique_rank_ != other.unique_rank_) {
    return false;
  }
  if (hashed_peers_.size() != other.hashed_peers_.size()) {
    return false;
  }
  for (auto it1 = hashed_peers_.begin(), it2 = other.hashed_peers_.begin();
       it1 != hashed_peers_.end(); ++it1, ++it2) {
    if (it1->link_hash != it2->link_hash) {
      return false;
    }
    if (it1->peer->unique_rank_ != it2->peer->unique_rank_) {
      return false;
    }
  }
  return true;
}
constexpr int Signature::kMaxGraphSize;
string Signature::ToString() const {
  string result;
  for (size_t n = 0; n < nodes.size(); ++n) {
    result += absl::StrFormat("%d:%s", n, nodes[n]->opcode());
    for (const auto& entry : nodes[n]->hashed_peers_) {
      const auto& link = nodes[n]->hash_to_link_[entry.link_hash];
      if (link.tag.local.IsInbound()) {
        result +=
            absl::StrFormat("[%s:%s:%d]", string(link.tag.local),
                            string(link.tag.remote), entry.peer->unique_rank_);
      }
    }
    result.push_back(',');
  }
  return result;
}
Status Signature::Compute() {
  if (map.size() > kMaxGraphSize) {
    return Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "A graph of %d nodes is too big for signature computation, "
            "the maximal supported node count is %d.",
            map.size(), kMaxGraphSize));
  }
  size_t next_node_id = 0;
  sig_short = 0;
  sig_full.resize(0);  
  PrepareNodes();
  FindUniqueHashes(&next_node_id);
  while (next_node_id < map.size()) {
    ComputeOneRound(next_node_id);
    FindUniqueHashes(&next_node_id);
  }
  OrderLinks();
  return absl::OkStatus();
}
void Signature::PrepareNodes() {
  nodes.resize(0);  
  int64_t mask = 1;
  for (const auto& entry : map) {
    SigNode* node = entry.second.get();
    node->last_hashed_nodes_ = node->node_mask_ = mask;
    mask <<= 1;
    node->unique_rank_ = ~0;
    node->hash_is_final_ = false;
    node->ComputeTopoHash0();
    if (node->GetHighTopoHash() <= map.size()) {
      node->ReHighTopoHash();
    }
    nodes.emplace_back(node);
  }
}
void Signature::FindUniqueHashes(size_t* next_node_id_p) {
  std::stable_sort(nodes.begin() + *next_node_id_p, nodes.end(),
                   SigNode::NodeOrderLess());
  bool found_unique = false;
  for (size_t n = *next_node_id_p; n < nodes.size(); ++n) {
    size_t cur_hash = nodes[n]->GetHighTopoHash();
    if (n + 1 < nodes.size() && nodes[n + 1]->GetHighTopoHash() == cur_hash) {
      for (++n;
           n + 1 < nodes.size() && nodes[n + 1]->GetHighTopoHash() == cur_hash;
           ++n) {
      }
      if (found_unique || n != nodes.size() - 1) {
        continue;
      }
    }
    found_unique = true;
    size_t id = (*next_node_id_p)++;
    nodes[n]->unique_rank_ = id;
    size_t last_hash = nodes[n]->GetHighTopoHash();
    CombineHash(last_hash, &sig_short);
    sig_full.push_back(last_hash);
    nodes[n]->topo_hash_.resize(1);
    nodes[n]->topo_hash_[0] = id + 1;  
    nodes[n]->hash_is_final_ = true;
    nodes[n]->last_hashed_nodes_ = nodes[n]->node_mask_;
    if (n != id) {
      std::swap(nodes[id], nodes[n]);
    }
  }
}
void Signature::ComputeOneRound(size_t next_node_id) {
  int debug_i = 0;
  for (auto it = nodes.begin() + next_node_id; it != nodes.end(); ++it) {
    auto node = *it;
    node->topo_hash_.resize(1);
    node->last_hashed_nodes_ = node->node_mask_;
    node->hash_is_final_ = false;
    if (debug) {
      LOG(INFO) << "DEBUG distance=" << 0 << " node " << debug_i++ << " "
                << node->name() << " mask=" << std::hex
                << node->last_hashed_nodes_;
    }
  }
  bool stop = false;
  for (int distance = 1; !stop; ++distance) {
    for (auto it = nodes.begin() + next_node_id; it != nodes.end(); ++it) {
      auto node = *it;
      if (node->hash_is_final_) {
        continue;
      }
      node->ComputeTopoHash(distance);
      if (node->GetHighTopoHash() <= nodes.size()) {
        node->ReHighTopoHash();
      }
    }
    stop = true;
    debug_i = 0;
    for (auto it = nodes.begin() + next_node_id; it != nodes.end(); ++it) {
      auto node = *it;
      if (debug) {
        LOG(INFO) << "DEBUG distance=" << distance << " node " << debug_i++
                  << " " << node->name() << " oldmask=" << std::hex
                  << node->last_hashed_nodes_ << " mask=" << std::hex
                  << node->next_hashed_nodes_;
      }
      if (node->last_hashed_nodes_ == node->next_hashed_nodes_) {
        node->hash_is_final_ = true;
      } else {
        node->last_hashed_nodes_ = node->next_hashed_nodes_;
        stop = false;
      }
    }
  }
}
void Signature::OrderLinks() {
  for (const auto& node : nodes) {
    if (node->hashed_peers_.empty()) {
      continue;
    }
    size_t cur_link_hash = node->hashed_peers_[0].link_hash + 1;
    int first_idx = -1;
    int idx;
    for (idx = 0; idx < static_cast<int64_t>(node->hashed_peers_.size());
         ++idx) {
      auto& entry = node->hashed_peers_[idx];
      if (entry.link_hash == cur_link_hash) {
        continue;
      }
      if (idx - first_idx > 1) {
        std::sort(node->hashed_peers_.begin() + first_idx,
                  node->hashed_peers_.begin() + idx,
                  SigNode::HashedPeer::LessByRank());
      }
      cur_link_hash = entry.link_hash;
      first_idx = idx;
    }
    if (idx - first_idx > 1) {
      std::sort(node->hashed_peers_.begin() + first_idx,
                node->hashed_peers_.begin() + idx,
                SigNode::HashedPeer::LessByRank());
    }
  }
}
bool Signature::operator==(const Signature& other) const {
  if (sig_short != other.sig_short) {
    return false;
  }
  if (sig_full.size() != other.sig_full.size()) {
    return false;
  }
  for (auto it1 = sig_full.begin(), it2 = other.sig_full.begin();
       it1 != sig_full.end(); ++it1, ++it2) {
    if (*it1 != *it2) {
      return false;
    }
  }
  if (nodes.size() != other.nodes.size()) {
    return false;
  }
  for (auto it1 = nodes.begin(), it2 = other.nodes.begin(); it1 != nodes.end();
       ++it1, ++it2) {
    if (**it1 != **it2) {
      return false;
    }
  }
  return true;
}
}  
}  
}  