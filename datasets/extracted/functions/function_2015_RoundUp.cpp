#ifndef SRC_LANGSVR_UTILS_BLOCK_ALLOCATOR_H_
#define SRC_LANGSVR_UTILS_BLOCK_ALLOCATOR_H_
#include <stdint.h>
#include <array>
#include <cstring>
#include <utility>
namespace langsvr {
template <typename T>
inline constexpr T RoundUp(T alignment, T value) {
    return ((value + alignment - 1) / alignment) * alignment;
}
template <typename TO, typename FROM>
inline TO Bitcast(FROM&& from) {
    static_assert(sizeof(FROM) == sizeof(TO));
    static_assert(std::is_trivially_copyable_v<std::decay_t<FROM>>);
    static_assert(std::is_trivially_copyable_v<std::decay_t<TO>>);
    TO to;
    memcpy(reinterpret_cast<std::byte*>(&to), reinterpret_cast<const std::byte*>(&from),
           sizeof(TO));
    return to;
}
template <typename T, size_t BLOCK_SIZE = 64 * 1024, size_t BLOCK_ALIGNMENT = 16>
class BlockAllocator {
    struct Pointers {
        static constexpr size_t kMax = 32;
        std::array<T*, kMax> ptrs;
        Pointers* next;
        Pointers* prev;
        size_t count;
    };
    struct alignas(BLOCK_ALIGNMENT) Block {
        uint8_t data[BLOCK_SIZE];
        Block* next = nullptr;
    };
    template <bool IS_CONST>
    class TView;
    template <bool IS_CONST>
    class TIterator {
        using PointerTy = std::conditional_t<IS_CONST, const T*, T*>;
      public:
        bool operator==(const TIterator& other) const {
            return ptrs == other.ptrs && idx == other.idx;
        }
        bool operator!=(const TIterator& other) const { return !(*this == other); }
        TIterator& operator++() {
            if (ptrs != nullptr) {
                ++idx;
                if (idx >= ptrs->count) {
                    idx = 0;
                    ptrs = ptrs->next;
                }
            }
            return *this;
        }
        TIterator& operator--() {
            if (ptrs != nullptr) {
                if (idx == 0) {
                    ptrs = ptrs->prev;
                    idx = ptrs->count - 1;
                }
                --idx;
            }
            return *this;
        }
        PointerTy operator*() const { return ptrs->ptrs[idx]; }
      private:
        friend TView<IS_CONST>;  
        explicit TIterator(const Pointers* p, size_t i) : ptrs(p), idx(i) {}
        const Pointers* ptrs = nullptr;
        size_t idx = 0;
    };
    template <bool IS_CONST>
    class TView {
      public:
        TIterator<IS_CONST> begin() const {
            return TIterator<IS_CONST>{allocator_->data.pointers.root, 0};
        }
        TIterator<IS_CONST> end() const { return TIterator<IS_CONST>{nullptr, 0}; }
      private:
        friend BlockAllocator;  
        explicit TView(BlockAllocator const* allocator) : allocator_(allocator) {}
        BlockAllocator const* const allocator_;
    };
  public:
    using Iterator = TIterator< false>;
    using ConstIterator = TIterator< true>;
    using View = TView<false>;
    using ConstView = TView<true>;
    BlockAllocator() = default;
    BlockAllocator(BlockAllocator&& rhs) { std::swap(data, rhs.data); }
    BlockAllocator& operator=(BlockAllocator&& rhs) {
        if (this != &rhs) {
            Reset();
            std::swap(data, rhs.data);
        }
        return *this;
    }
    ~BlockAllocator() { Reset(); }
    View Objects() { return View(this); }
    ConstView Objects() const { return ConstView(this); }
    template <typename TYPE = T, typename... ARGS>
    TYPE* Create(ARGS&&... args) {
        static_assert(std::is_same<T, TYPE>::value || std::is_base_of<T, TYPE>::value,
                      "TYPE does not derive from T");
        static_assert(std::is_same<T, TYPE>::value || std::has_virtual_destructor<T>::value,
                      "TYPE requires a virtual destructor when calling Create() for a type "
                      "that is not T");
        auto* ptr = Allocate<TYPE>();
        new (ptr) TYPE(std::forward<ARGS>(args)...);
        AddObjectPointer(ptr);
        data.count++;
        return ptr;
    }
    void Reset() {
        for (auto ptr : Objects()) {
            ptr->~T();
        }
        auto* block = data.block.root;
        while (block != nullptr) {
            auto* next = block->next;
            delete block;
            block = next;
        }
        data = {};
    }
    size_t Count() const { return data.count; }
  private:
    BlockAllocator(const BlockAllocator&) = delete;
    BlockAllocator& operator=(const BlockAllocator&) = delete;
    template <typename TYPE>
    TYPE* Allocate() {
        static_assert(sizeof(TYPE) <= BLOCK_SIZE,
                      "Cannot construct TYPE with size greater than BLOCK_SIZE");
        static_assert(alignof(TYPE) <= BLOCK_ALIGNMENT, "alignof(TYPE) is greater than ALIGNMENT");
        auto& block = data.block;
        block.current_offset = RoundUp(alignof(TYPE), block.current_offset);
        if (block.current_offset + sizeof(TYPE) > BLOCK_SIZE) {
            auto* prev_block = block.current;
            block.current = new Block;
            if (!block.current) {
                return nullptr;  
            }
            block.current->next = nullptr;
            block.current_offset = 0;
            if (prev_block) {
                prev_block->next = block.current;
            } else {
                block.root = block.current;
            }
        }
        auto* base = &block.current->data[0];
        auto* ptr = Bitcast<TYPE*>(base + block.current_offset);
        block.current_offset += sizeof(TYPE);
        return ptr;
    }
    void AddObjectPointer(T* ptr) {
        auto& pointers = data.pointers;
        if (!pointers.current || pointers.current->count == Pointers::kMax) {
            auto* prev_pointers = pointers.current;
            pointers.current = Allocate<Pointers>();
            if (!pointers.current) {
                return;  
            }
            pointers.current->next = nullptr;
            pointers.current->prev = prev_pointers;
            pointers.current->count = 0;
            if (prev_pointers) {
                prev_pointers->next = pointers.current;
            } else {
                pointers.root = pointers.current;
            }
        }
        pointers.current->ptrs[pointers.current->count++] = ptr;
    }
    struct {
        struct {
            Block* root = nullptr;
            Block* current = nullptr;
            size_t current_offset = BLOCK_SIZE;
        } block;
        struct {
            Pointers* root = nullptr;
            Pointers* current = nullptr;
        } pointers;
        size_t count = 0;
    } data;
};
}  
#endif  