#include "tensorflow/lite/experimental/acceleration/mini_benchmark/big_little_affinity.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "include/cpuinfo.h"

namespace tflite {
namespace acceleration {
namespace {

struct FakeCluster {
  cpuinfo_cluster cluster{};
};

struct FakeCore {
  cpuinfo_core core{};
};

struct FakeProcessor {
  cpuinfo_processor processor{};
  FakeCore core_storage{};
  FakeCluster cluster_storage{};
};

std::vector<FakeProcessor> g_processors;
bool g_cpuinfo_initialize_result = true;

void SetFakeProcessors(
    std::initializer_list<std::tuple<uint32_t, uint32_t, uint64_t,
                                     cpuinfo_uarch>>
        processors) {
  g_processors.clear();

  for (const auto& [linux_id, cluster_id, frequency, uarch] : processors) {
    FakeProcessor fake{};

    fake.cluster_storage.cluster.cluster_id = cluster_id;

    fake.core_storage.core.frequency = frequency;
    fake.core_storage.core.uarch = uarch;

    fake.processor.linux_id = linux_id;
    fake.processor.core = &fake.core_storage.core;
    fake.processor.cluster = &fake.cluster_storage.cluster;

    g_processors.push_back(fake);
  }
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite

extern "C" {

bool cpuinfo_initialize(void) { return tflite::acceleration::g_cpuinfo_initialize_result; }

uint32_t cpuinfo_get_processors_count(void) {
  return static_cast<uint32_t>(tflite::acceleration::g_processors.size());
}

const struct cpuinfo_processor* cpuinfo_get_processor(uint32_t index) {
  return &tflite::acceleration::g_processors[index].processor;
}

}  // extern "C"

namespace tflite {
namespace acceleration {
namespace {

class BigLittleAffinityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_cpuinfo_initialize_result = true;
    g_processors.clear();
  }

  void TearDown() override {
    g_cpuinfo_initialize_result = true;
    g_processors.clear();
  }
};

TEST_F(BigLittleAffinityTest, Invalid_CpuinfoInitializeFailsReturnsDefaultAffinity) {
  g_cpuinfo_initialize_result = false;

  BigLittleAffinity affinity = GetAffinity();

  EXPECT_EQ(affinity.big_core_affinity, 0u);
  EXPECT_EQ(affinity.little_core_affinity, 0u);
}

TEST_F(BigLittleAffinityTest, BVA_NoProcessorsReturnsDefaultAffinity) {
  SetFakeProcessors({});

  BigLittleAffinity affinity = GetAffinity();

  EXPECT_EQ(affinity.big_core_affinity, 0u);
  EXPECT_EQ(affinity.little_core_affinity, 0u);
}

TEST_F(BigLittleAffinityTest, ECP_SingleClusterSameFrequencyMirrorsAffinity) {
  SetFakeProcessors({
      {0, 0, 1800000, cpuinfo_uarch_cortex_a53},
      {1, 0, 1800000, cpuinfo_uarch_cortex_a53},
      {2, 0, 1800000, cpuinfo_uarch_cortex_a53},
      {3, 0, 1800000, cpuinfo_uarch_cortex_a53},
  });

  BigLittleAffinity affinity = GetAffinity();

#ifdef __ANDROID__
  EXPECT_EQ(affinity.big_core_affinity, affinity.little_core_affinity);
  EXPECT_NE(affinity.big_core_affinity, 0u);
#else
  EXPECT_EQ(affinity.big_core_affinity, 0u);
  EXPECT_EQ(affinity.little_core_affinity, 0u);
#endif
}

TEST_F(BigLittleAffinityTest, ECP_AllProcessorsSameMaxFrequencySameArchMirrorsAffinity) {
  SetFakeProcessors({
      {0, 0, 2000000, cpuinfo_uarch_cortex_a76},
      {1, 1, 2000000, cpuinfo_uarch_cortex_a76},
      {2, 2, 2000000, cpuinfo_uarch_cortex_a76},
      {3, 3, 2000000, cpuinfo_uarch_cortex_a76},
  });

  BigLittleAffinity affinity = GetAffinity();

#ifdef __ANDROID__
  EXPECT_EQ(affinity.big_core_affinity, affinity.little_core_affinity);
  EXPECT_NE(affinity.big_core_affinity, 0u);
#else
  EXPECT_EQ(affinity.big_core_affinity, 0u);
  EXPECT_EQ(affinity.little_core_affinity, 0u);
#endif
}

TEST_F(BigLittleAffinityTest, ECP_AllProcessorsSameFrequencyDifferentArchNotMirroredByArchRule) {
  SetFakeProcessors({
      {0, 0, 1800000, cpuinfo_uarch_cortex_a53},
      {1, 1, 1800000, cpuinfo_uarch_cortex_a76},
  });

  BigLittleAffinity affinity = GetAffinity();

#ifdef __ANDROID__
  EXPECT_NE(affinity.big_core_affinity, 0u);
  EXPECT_NE(affinity.little_core_affinity, 0u);
#else
  EXPECT_EQ(affinity.big_core_affinity, 0u);
  EXPECT_EQ(affinity.little_core_affinity, 0u);
#endif
}

TEST_F(BigLittleAffinityTest, ECP_TwoLargestFrequencyProcessorsClassifiesOnlyThemAsBig) {
  SetFakeProcessors({
      {0, 0, 1200000, cpuinfo_uarch_cortex_a53},
      {1, 0, 1200000, cpuinfo_uarch_cortex_a53},
      {2, 1, 2500000, cpuinfo_uarch_cortex_a76},
      {3, 1, 2500000, cpuinfo_uarch_cortex_a76},
  });

  BigLittleAffinity affinity = GetAffinity();

#ifdef __ANDROID__
  EXPECT_EQ(affinity.little_core_affinity, 0x3u);
  EXPECT_EQ(affinity.big_core_affinity, 0xCu);
#else
  EXPECT_EQ(affinity.big_core_affinity, 0u);
  EXPECT_EQ(affinity.little_core_affinity, 0u);
#endif
}

TEST_F(BigLittleAffinityTest, ECP_MoreThanTwoLargestFrequencyProcessorsUsesSmallestAsLittle) {
  SetFakeProcessors({
      {0, 0, 1000000, cpuinfo_uarch_cortex_a53},
      {1, 0, 1000000, cpuinfo_uarch_cortex_a53},
      {2, 1, 2000000, cpuinfo_uarch_cortex_a76},
      {3, 1, 2000000, cpuinfo_uarch_cortex_a76},
      {4, 1, 2000000, cpuinfo_uarch_cortex_a76},
  });

  BigLittleAffinity affinity = GetAffinity();

#ifdef __ANDROID__
  EXPECT_EQ(affinity.little_core_affinity, 0x3u);
  EXPECT_EQ(affinity.big_core_affinity, 0x1Cu);
#else
  EXPECT_EQ(affinity.big_core_affinity, 0u);
  EXPECT_EQ(affinity.little_core_affinity, 0u);
#endif
}

TEST_F(BigLittleAffinityTest, BVA_ExactlyOneBigCoreWhenLargestCountIsOneUsesSmallestAsLittle) {
  SetFakeProcessors({
      {0, 0, 1200000, cpuinfo_uarch_cortex_a53},
      {1, 0, 1200000, cpuinfo_uarch_cortex_a53},
      {2, 1, 2500000, cpuinfo_uarch_cortex_a76},
  });

  BigLittleAffinity affinity = GetAffinity();

#ifdef __ANDROID__
  EXPECT_EQ(affinity.little_core_affinity, 0x3u);
  EXPECT_EQ(affinity.big_core_affinity, 0x4u);
#else
  EXPECT_EQ(affinity.big_core_affinity, 0u);
  EXPECT_EQ(affinity.little_core_affinity, 0u);
#endif
}

TEST_F(BigLittleAffinityTest, BVA_ExactlyTwoBigCoresUsesSpecialCase) {
  SetFakeProcessors({
      {0, 0, 1000000, cpuinfo_uarch_cortex_a53},
      {1, 1, 1500000, cpuinfo_uarch_cortex_a55},
      {2, 2, 2500000, cpuinfo_uarch_cortex_a76},
      {3, 2, 2500000, cpuinfo_uarch_cortex_a76},
  });

  BigLittleAffinity affinity = GetAffinity();

#ifdef __ANDROID__
  EXPECT_EQ(affinity.little_core_affinity, 0x3u);
  EXPECT_EQ(affinity.big_core_affinity, 0xCu);
#else
  EXPECT_EQ(affinity.big_core_affinity, 0u);
  EXPECT_EQ(affinity.little_core_affinity, 0u);
#endif
}

TEST_F(BigLittleAffinityTest, Edge_MiddleFrequencyClusterIsBigWhenLargestCountIsNotTwo) {
  SetFakeProcessors({
      {0, 0, 1000000, cpuinfo_uarch_cortex_a53},
      {1, 1, 1500000, cpuinfo_uarch_cortex_a55},
      {2, 2, 2500000, cpuinfo_uarch_cortex_a76},
      {3, 2, 2500000, cpuinfo_uarch_cortex_a76},
      {4, 2, 2500000, cpuinfo_uarch_cortex_a76},
  });

  BigLittleAffinity affinity = GetAffinity();

#ifdef __ANDROID__
  EXPECT_EQ(affinity.little_core_affinity, 0x1u);
  EXPECT_EQ(affinity.big_core_affinity, 0x1Eu);
#else
  EXPECT_EQ(affinity.big_core_affinity, 0u);
  EXPECT_EQ(affinity.little_core_affinity, 0u);
#endif
}

TEST_F(BigLittleAffinityTest, Edge_InOrderArchitecturesAreLittleWhenAllFrequenciesEqual) {
  SetFakeProcessors({
      {0, 0, 2000000, cpuinfo_uarch_cortex_a53},
      {1, 1, 2000000, cpuinfo_uarch_cortex_a55},
      {2, 2, 2000000, cpuinfo_uarch_cortex_a57},
      {3, 3, 2000000, cpuinfo_uarch_cortex_a76},
  });

  BigLittleAffinity affinity = GetAffinity();

#ifdef __ANDROID__
  EXPECT_EQ(affinity.little_core_affinity, 0x7u);
  EXPECT_EQ(affinity.big_core_affinity, 0x8u);
#else
  EXPECT_EQ(affinity.big_core_affinity, 0u);
  EXPECT_EQ(affinity.little_core_affinity, 0u);
#endif
}

TEST_F(BigLittleAffinityTest, Edge_CortexA55R0IsTreatedAsInOrderLittleCore) {
  SetFakeProcessors({
      {0, 0, 2000000, cpuinfo_uarch_cortex_a55r0},
      {1, 1, 2000000, cpuinfo_uarch_cortex_a76},
  });

  BigLittleAffinity affinity = GetAffinity();

#ifdef __ANDROID__
  EXPECT_EQ(affinity.little_core_affinity, 0x1u);
  EXPECT_EQ(affinity.big_core_affinity, 0x2u);
#else
  EXPECT_EQ(affinity.big_core_affinity, 0u);
  EXPECT_EQ(affinity.little_core_affinity, 0u);
#endif
}

TEST_F(BigLittleAffinityTest, Edge_ZeroFrequencyProcessorsDoNotContributeToClusterFrequencyMap) {
  SetFakeProcessors({
      {0, 0, 0, cpuinfo_uarch_cortex_a53},
      {1, 1, 2000000, cpuinfo_uarch_cortex_a76},
  });

  BigLittleAffinity affinity = GetAffinity();

#ifdef __ANDROID__
  EXPECT_EQ(affinity.big_core_affinity, affinity.little_core_affinity);
#else
  EXPECT_EQ(affinity.big_core_affinity, 0u);
  EXPECT_EQ(affinity.little_core_affinity, 0u);
#endif
}

TEST_F(BigLittleAffinityTest, BVA_MaxFrequencyUint64HandledAsLargestFrequency) {
  SetFakeProcessors({
      {0, 0, 1, cpuinfo_uarch_cortex_a53},
      {1, 1, UINT64_MAX, cpuinfo_uarch_cortex_a76},
  });

  BigLittleAffinity affinity = GetAffinity();

#ifdef __ANDROID__
  EXPECT_EQ(affinity.little_core_affinity, 0x1u);
  EXPECT_EQ(affinity.big_core_affinity, 0x2u);
#else
  EXPECT_EQ(affinity.big_core_affinity, 0u);
  EXPECT_EQ(affinity.little_core_affinity, 0u);
#endif
}

TEST_F(BigLittleAffinityTest, BVA_LinuxIdZeroSetsLowestAffinityBit) {
  SetFakeProcessors({
      {0, 0, 1000000, cpuinfo_uarch_cortex_a53},
      {1, 1, 2000000, cpuinfo_uarch_cortex_a76},
  });

  BigLittleAffinity affinity = GetAffinity();

#ifdef __ANDROID__
  EXPECT_TRUE((affinity.little_core_affinity & 0x1u) != 0u);
  EXPECT_TRUE((affinity.big_core_affinity & 0x2u) != 0u);
#else
  EXPECT_EQ(affinity.big_core_affinity, 0u);
  EXPECT_EQ(affinity.little_core_affinity, 0u);
#endif
}

TEST_F(BigLittleAffinityTest, Edge_NonSequentialLinuxIdsSetExpectedAffinityBits) {
  SetFakeProcessors({
      {2, 0, 1000000, cpuinfo_uarch_cortex_a53},
      {5, 1, 2000000, cpuinfo_uarch_cortex_a76},
  });

  BigLittleAffinity affinity = GetAffinity();

#ifdef __ANDROID__
  EXPECT_EQ(affinity.little_core_affinity, 0x4u);
  EXPECT_EQ(affinity.big_core_affinity, 0x20u);
#else
  EXPECT_EQ(affinity.big_core_affinity, 0u);
  EXPECT_EQ(affinity.little_core_affinity, 0u);
#endif
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite