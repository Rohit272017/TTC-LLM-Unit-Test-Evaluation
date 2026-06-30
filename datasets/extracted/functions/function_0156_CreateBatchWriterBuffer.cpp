#include "quiche/quic/core/batch_writer/quic_gso_batch_writer.h"
#include <time.h>
#include <ctime>
#include <memory>
#include <utility>
#include "quiche/quic/core/quic_linux_socket_utils.h"
#include "quiche/quic/platform/api/quic_server_stats.h"
namespace quic {
std::unique_ptr<QuicBatchWriterBuffer>
QuicGsoBatchWriter::CreateBatchWriterBuffer() {
  return std::make_unique<QuicBatchWriterBuffer>();
}
QuicGsoBatchWriter::QuicGsoBatchWriter(int fd)
    : QuicGsoBatchWriter(fd, CLOCK_MONOTONIC) {}
QuicGsoBatchWriter::QuicGsoBatchWriter(int fd,
                                       clockid_t clockid_for_release_time)
    : QuicUdpBatchWriter(CreateBatchWriterBuffer(), fd),
      clockid_for_release_time_(clockid_for_release_time),
      supports_release_time_(
          GetQuicRestartFlag(quic_support_release_time_for_gso) &&
          QuicLinuxSocketUtils::EnableReleaseTime(fd,
                                                  clockid_for_release_time)) {
  if (supports_release_time_) {
    QUIC_RESTART_FLAG_COUNT(quic_support_release_time_for_gso);
  }
}
QuicGsoBatchWriter::QuicGsoBatchWriter(
    std::unique_ptr<QuicBatchWriterBuffer> batch_buffer, int fd,
    clockid_t clockid_for_release_time, ReleaseTimeForceEnabler )
    : QuicUdpBatchWriter(std::move(batch_buffer), fd),
      clockid_for_release_time_(clockid_for_release_time),
      supports_release_time_(true) {
  QUIC_DLOG(INFO) << "Release time forcefully enabled.";
}
QuicGsoBatchWriter::CanBatchResult QuicGsoBatchWriter::CanBatch(
    const char* , size_t buf_len, const QuicIpAddress& self_address,
    const QuicSocketAddress& peer_address, const PerPacketOptions* ,
    const QuicPacketWriterParams& params, uint64_t release_time) const {
  if (buffered_writes().empty()) {
    return CanBatchResult(true, false);
  }
  const BufferedWrite& first = buffered_writes().front();
  const BufferedWrite& last = buffered_writes().back();
  const bool can_burst = !SupportsReleaseTime() ||
                         params.release_time_delay.IsZero() ||
                         params.allow_burst;
  size_t max_segments = MaxSegments(first.buf_len);
  bool can_batch =
      buffered_writes().size() < max_segments &&                    
      last.self_address == self_address &&                          
      last.peer_address == peer_address &&                          
      batch_buffer().SizeInUse() + buf_len <= kMaxGsoPacketSize &&  
      first.buf_len == last.buf_len &&                              
      first.buf_len >= buf_len &&                                   
      first.params.ecn_codepoint == params.ecn_codepoint &&         
      (can_burst || first.release_time == release_time);            
  bool must_flush = (!can_batch) ||                                  
                    (last.buf_len != buf_len) ||                     
                    (buffered_writes().size() + 1 == max_segments);  
  return CanBatchResult(can_batch, must_flush);
}
QuicGsoBatchWriter::ReleaseTime QuicGsoBatchWriter::GetReleaseTime(
    const QuicPacketWriterParams& params) const {
  QUICHE_DCHECK(SupportsReleaseTime());
  const uint64_t now = NowInNanosForReleaseTime();
  const uint64_t ideal_release_time =
      now + params.release_time_delay.ToMicroseconds() * 1000;
  if ((params.release_time_delay.IsZero() || params.allow_burst) &&
      !buffered_writes().empty() &&
      (buffered_writes().back().release_time >= now)) {
    const uint64_t actual_release_time = buffered_writes().back().release_time;
    const int64_t offset_ns = actual_release_time - ideal_release_time;
    ReleaseTime result{actual_release_time,
                       QuicTime::Delta::FromMicroseconds(offset_ns / 1000)};
    QUIC_DVLOG(1) << "ideal_release_time:" << ideal_release_time
                  << ", actual_release_time:" << actual_release_time
                  << ", offset:" << result.release_time_offset;
    return result;
  }
  return {ideal_release_time, QuicTime::Delta::Zero()};
}
uint64_t QuicGsoBatchWriter::NowInNanosForReleaseTime() const {
  struct timespec ts;
  if (clock_gettime(clockid_for_release_time_, &ts) != 0) {
    return 0;
  }
  return ts.tv_sec * (1000ULL * 1000 * 1000) + ts.tv_nsec;
}
void QuicGsoBatchWriter::BuildCmsg(QuicMsgHdr* hdr,
                                   const QuicIpAddress& self_address,
                                   uint16_t gso_size, uint64_t release_time,
                                   QuicEcnCodepoint ecn_codepoint) {
  hdr->SetIpInNextCmsg(self_address);
  if (gso_size > 0) {
    *hdr->GetNextCmsgData<uint16_t>(SOL_UDP, UDP_SEGMENT) = gso_size;
  }
  if (release_time != 0) {
    *hdr->GetNextCmsgData<uint64_t>(SOL_SOCKET, SO_TXTIME) = release_time;
  }
  if (ecn_codepoint != ECN_NOT_ECT && GetQuicRestartFlag(quic_support_ect1)) {
    QUIC_RESTART_FLAG_COUNT_N(quic_support_ect1, 8, 9);
    if (self_address.IsIPv4()) {
      *hdr->GetNextCmsgData<int>(IPPROTO_IP, IP_TOS) =
          static_cast<int>(ecn_codepoint);
    } else {
      *hdr->GetNextCmsgData<int>(IPPROTO_IPV6, IPV6_TCLASS) =
          static_cast<int>(ecn_codepoint);
    }
  }
}
QuicGsoBatchWriter::FlushImplResult QuicGsoBatchWriter::FlushImpl() {
  return InternalFlushImpl<kCmsgSpace>(BuildCmsg);
}
}  