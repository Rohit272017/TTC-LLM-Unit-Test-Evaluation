#include "common/memory.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <ostream>
#include <sstream>

#include "google/protobuf/arena.h"

namespace cel {
namespace {

bool IsAligned(void* ptr, size_t alignment) {
  return reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0;
}

TEST(MemoryManagementStreamTest, PrintsPooling) {
  std::ostringstream os;

  os << MemoryManagement::kPooling;

  EXPECT_EQ(os.str(), "POOLING");
}

TEST(MemoryManagementStreamTest, PrintsReferenceCounting) {
  std::ostringstream os;

  os << MemoryManagement::kReferenceCounting;

  EXPECT_EQ(os.str(), "REFERENCE_COUNTING");
}

TEST(ReferenceCountingMemoryManagerTest, AllocateZeroSizeReturnsNull) {
  ReferenceCountingMemoryManager manager;

  void* ptr = manager.Allocate(0, alignof(std::max_align_t));

  EXPECT_EQ(ptr, nullptr);
}

TEST(ReferenceCountingMemoryManagerTest, DeallocateNullPointerWithZeroSizeReturnsFalse) {
  ReferenceCountingMemoryManager manager;

  bool deallocated = manager.Deallocate(nullptr, 0, alignof(std::max_align_t));

  EXPECT_FALSE(deallocated);
}

TEST(ReferenceCountingMemoryManagerTest, AllocateOneByteWithDefaultAlignmentReturnsAlignedPointer) {
  ReferenceCountingMemoryManager manager;

  void* ptr = manager.Allocate(1, alignof(std::max_align_t));

  ASSERT_NE(ptr, nullptr);
  EXPECT_TRUE(IsAligned(ptr, alignof(std::max_align_t)));

  EXPECT_TRUE(manager.Deallocate(ptr, 1, alignof(std::max_align_t)));
}

TEST(ReferenceCountingMemoryManagerTest, AllocateMultipleBytesWithDefaultAlignmentReturnsAlignedPointer) {
  ReferenceCountingMemoryManager manager;
  constexpr size_t kSize = 128;
  constexpr size_t kAlignment = alignof(std::max_align_t);

  void* ptr = manager.Allocate(kSize, kAlignment);

  ASSERT_NE(ptr, nullptr);
  EXPECT_TRUE(IsAligned(ptr, kAlignment));

  EXPECT_TRUE(manager.Deallocate(ptr, kSize, kAlignment));
}

TEST(ReferenceCountingMemoryManagerTest, AllocateWithAlignmentOneIsAccepted) {
  ReferenceCountingMemoryManager manager;
  constexpr size_t kSize = 16;
  constexpr size_t kAlignment = 1;

  void* ptr = manager.Allocate(kSize, kAlignment);

  ASSERT_NE(ptr, nullptr);
  EXPECT_TRUE(IsAligned(ptr, kAlignment));

  EXPECT_TRUE(manager.Deallocate(ptr, kSize, kAlignment));
}

TEST(ReferenceCountingMemoryManagerTest, AllocateWithAlignmentTwoIsAccepted) {
  ReferenceCountingMemoryManager manager;
  constexpr size_t kSize = 16;
  constexpr size_t kAlignment = 2;

  void* ptr = manager.Allocate(kSize, kAlignment);

  ASSERT_NE(ptr, nullptr);
  EXPECT_TRUE(IsAligned(ptr, kAlignment));

  EXPECT_TRUE(manager.Deallocate(ptr, kSize, kAlignment));
}

TEST(ReferenceCountingMemoryManagerTest, AllocateWithAlignmentEqualToDefaultNewAlignment) {
  ReferenceCountingMemoryManager manager;
  constexpr size_t kSize = 64;
  constexpr size_t kAlignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;

  void* ptr = manager.Allocate(kSize, kAlignment);

  ASSERT_NE(ptr, nullptr);
  EXPECT_TRUE(IsAligned(ptr, kAlignment));

  EXPECT_TRUE(manager.Deallocate(ptr, kSize, kAlignment));
}

TEST(ReferenceCountingMemoryManagerTest, AllocateWithAlignmentGreaterThanDefaultNewAlignment) {
  ReferenceCountingMemoryManager manager;
  constexpr size_t kSize = 64;
  constexpr size_t kAlignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2;

  void* ptr = manager.Allocate(kSize, kAlignment);

  ASSERT_NE(ptr, nullptr);
  EXPECT_TRUE(IsAligned(ptr, kAlignment));

  EXPECT_TRUE(manager.Deallocate(ptr, kSize, kAlignment));
}

TEST(ReferenceCountingMemoryManagerTest, AllocateWithLargeOverAlignment) {
  ReferenceCountingMemoryManager manager;
  constexpr size_t kSize = 256;
  constexpr size_t kAlignment = 64;

  void* ptr = manager.Allocate(kSize, kAlignment);

  ASSERT_NE(ptr, nullptr);
  EXPECT_TRUE(IsAligned(ptr, kAlignment));

  EXPECT_TRUE(manager.Deallocate(ptr, kSize, kAlignment));
}

TEST(ReferenceCountingMemoryManagerTest, AllocateZeroSizeWithOverAlignmentReturnsNull) {
  ReferenceCountingMemoryManager manager;
  constexpr size_t kAlignment = 64;

  void* ptr = manager.Allocate(0, kAlignment);

  EXPECT_EQ(ptr, nullptr);
}

TEST(ReferenceCountingMemoryManagerTest, DeallocateNullPointerWithOverAlignmentReturnsFalse) {
  ReferenceCountingMemoryManager manager;
  constexpr size_t kAlignment = 64;

  bool deallocated = manager.Deallocate(nullptr, 0, kAlignment);

  EXPECT_FALSE(deallocated);
}

TEST(ReferenceCountingMemoryManagerTest, AllocateAndDeallocateSeveralSizes) {
  ReferenceCountingMemoryManager manager;

  for (size_t size : {1u, 2u, 7u, 8u, 15u, 16u, 31u, 32u, 1024u}) {
    void* ptr = manager.Allocate(size, alignof(std::max_align_t));

    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(IsAligned(ptr, alignof(std::max_align_t)));
    EXPECT_TRUE(manager.Deallocate(ptr, size, alignof(std::max_align_t)));
  }
}

TEST(ReferenceCountingMemoryManagerTest, AllocateAndDeallocateSeveralOverAlignedSizes) {
  ReferenceCountingMemoryManager manager;
  constexpr size_t kAlignment = 64;

  for (size_t size : {1u, 2u, 7u, 8u, 15u, 16u, 31u, 32u, 1024u}) {
    void* ptr = manager.Allocate(size, kAlignment);

    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(IsAligned(ptr, kAlignment));
    EXPECT_TRUE(manager.Deallocate(ptr, size, kAlignment));
  }
}

TEST(ReferenceCountingMemoryManagerTest, AllocatedMemoryCanBeWrittenAndRead) {
  ReferenceCountingMemoryManager manager;
  constexpr size_t kSize = 32;
  constexpr size_t kAlignment = alignof(std::max_align_t);

  void* ptr = manager.Allocate(kSize, kAlignment);

  ASSERT_NE(ptr, nullptr);

  auto* bytes = static_cast<unsigned char*>(ptr);
  for (size_t i = 0; i < kSize; ++i) {
    bytes[i] = static_cast<unsigned char>(i);
  }

  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_EQ(bytes[i], static_cast<unsigned char>(i));
  }

  EXPECT_TRUE(manager.Deallocate(ptr, kSize, kAlignment));
}

TEST(ReferenceCountingMemoryManagerTest, OverAlignedAllocatedMemoryCanBeWrittenAndRead) {
  ReferenceCountingMemoryManager manager;
  constexpr size_t kSize = 128;
  constexpr size_t kAlignment = 64;

  void* ptr = manager.Allocate(kSize, kAlignment);

  ASSERT_NE(ptr, nullptr);
  ASSERT_TRUE(IsAligned(ptr, kAlignment));

  auto* bytes = static_cast<unsigned char*>(ptr);
  for (size_t i = 0; i < kSize; ++i) {
    bytes[i] = static_cast<unsigned char>(255u - i);
  }

  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_EQ(bytes[i], static_cast<unsigned char>(255u - i));
  }

  EXPECT_TRUE(manager.Deallocate(ptr, kSize, kAlignment));
}

TEST(MemoryManagerTest, UnmanagedReturnsPoolingMemoryManager) {
  MemoryManager manager = MemoryManager::Unmanaged();

  EXPECT_EQ(manager.memory_management(), MemoryManagement::kPooling);
}

TEST(MemoryManagerTest, UnmanagedCanBeCalledMultipleTimes) {
  MemoryManager manager1 = MemoryManager::Unmanaged();
  MemoryManager manager2 = MemoryManager::Unmanaged();

  EXPECT_EQ(manager1.memory_management(), MemoryManagement::kPooling);
  EXPECT_EQ(manager2.memory_management(), MemoryManagement::kPooling);
}

#ifndef NDEBUG

TEST(ReferenceCountingMemoryManagerDeathTest, AllocateWithZeroAlignmentFailsDcheck) {
  ReferenceCountingMemoryManager manager;

  EXPECT_DEATH(static_cast<void>(manager.Allocate(1, 0)), "");
}

TEST(ReferenceCountingMemoryManagerDeathTest, AllocateWithNonPowerOfTwoAlignmentFailsDcheck) {
  ReferenceCountingMemoryManager manager;

  EXPECT_DEATH(static_cast<void>(manager.Allocate(1, 3)), "");
}

TEST(ReferenceCountingMemoryManagerDeathTest, DeallocateWithZeroAlignmentFailsDcheck) {
  ReferenceCountingMemoryManager manager;

  EXPECT_DEATH(static_cast<void>(manager.Deallocate(nullptr, 0, 0)), "");
}

TEST(ReferenceCountingMemoryManagerDeathTest, DeallocateWithNonPowerOfTwoAlignmentFailsDcheck) {
  ReferenceCountingMemoryManager manager;

  EXPECT_DEATH(static_cast<void>(manager.Deallocate(nullptr, 0, 6)), "");
}

TEST(ReferenceCountingMemoryManagerDeathTest, DeallocateNullPointerWithNonZeroSizeFailsDcheck) {
  ReferenceCountingMemoryManager manager;

  EXPECT_DEATH(static_cast<void>(
                   manager.Deallocate(nullptr, 1, alignof(std::max_align_t))),
               "");
}

#endif  // NDEBUG

}  // namespace
}  // namespace cel