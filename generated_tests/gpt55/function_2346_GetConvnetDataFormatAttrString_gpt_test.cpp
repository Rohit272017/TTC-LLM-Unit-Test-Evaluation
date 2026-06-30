#include "tensorflow/core/util/tensor_format.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

namespace tensorflow {
namespace {

TEST(TensorFormatTest, AttrStringConvnet2D) {
  EXPECT_EQ(GetConvnetDataFormatAttrString(),
            "data_format: { 'NHWC', 'NCHW' } = 'NHWC' ");
}

TEST(TensorFormatTest, AttrStringConvnet3D) {
  EXPECT_EQ(GetConvnet3dDataFormatAttrString(),
            "data_format: { 'NDHWC', 'NCDHW' } = 'NDHWC' ");
}

TEST(TensorFormatTest, AttrStringConvnet2D3D) {
  EXPECT_EQ(GetConvnetDataFormat2D3DAttrString(),
            "data_format: { 'NHWC', 'NCHW', 'NDHWC', 'NCDHW' } = 'NHWC' ");
}

TEST(TensorFormatTest, AttrStringFilter2D) {
  EXPECT_EQ(GetConvnetFilterFormatAttrString(),
            "filter_format: { 'HWIO', 'OIHW' } = 'HWIO' ");
}

TEST(TensorFormatTest, AttrStringFilter3D) {
  EXPECT_EQ(GetConvnet3dFilterFormatAttrString(),
            "filter_format: { 'DHWIO', 'OIDHW' } = 'DHWIO' ");
}

TEST(TensorFormatTest, ECP_ToStringTensorFormats) {
  EXPECT_EQ(ToString(FORMAT_NHWC), "NHWC");
  EXPECT_EQ(ToString(FORMAT_NCHW), "NCHW");
  EXPECT_EQ(ToString(FORMAT_NCHW_VECT_C), "NCHW_VECT_C");
  EXPECT_EQ(ToString(FORMAT_NHWC_VECT_W), "NHWC_VECT_W");
  EXPECT_EQ(ToString(FORMAT_HWNC), "HWNC");
  EXPECT_EQ(ToString(FORMAT_HWCN), "HWCN");
}

TEST(TensorFormatTest, ECP_ToStringFilterTensorFormats) {
  EXPECT_EQ(ToString(FORMAT_HWIO), "HWIO");
  EXPECT_EQ(ToString(FORMAT_OIHW), "OIHW");
  EXPECT_EQ(ToString(FORMAT_OHWI), "OHWI");
  EXPECT_EQ(ToString(FORMAT_OIHW_VECT_I), "OIHW_VECT_I");
}

TEST(TensorFormatTest, ECP_FormatFromString2DAnd3DAliases) {
  TensorFormat format = FORMAT_NHWC;

  EXPECT_TRUE(FormatFromString("NHWC", &format));
  EXPECT_EQ(format, FORMAT_NHWC);

  format = FORMAT_NCHW;
  EXPECT_TRUE(FormatFromString("NDHWC", &format));
  EXPECT_EQ(format, FORMAT_NHWC);

  format = FORMAT_NHWC;
  EXPECT_TRUE(FormatFromString("NCHW", &format));
  EXPECT_EQ(format, FORMAT_NCHW);

  format = FORMAT_NHWC;
  EXPECT_TRUE(FormatFromString("NCDHW", &format));
  EXPECT_EQ(format, FORMAT_NCHW);
}

TEST(TensorFormatTest, ECP_FormatFromStringVectorAndSpecialFormats) {
  TensorFormat format = FORMAT_NHWC;

  EXPECT_TRUE(FormatFromString("NCHW_VECT_C", &format));
  EXPECT_EQ(format, FORMAT_NCHW_VECT_C);

  EXPECT_TRUE(FormatFromString("NHWC_VECT_W", &format));
  EXPECT_EQ(format, FORMAT_NHWC_VECT_W);

  EXPECT_TRUE(FormatFromString("HWNC", &format));
  EXPECT_EQ(format, FORMAT_HWNC);

  EXPECT_TRUE(FormatFromString("HWCN", &format));
  EXPECT_EQ(format, FORMAT_HWCN);
}

TEST(TensorFormatTest, ECP_FilterFormatFromString2DAnd3DAliases) {
  FilterTensorFormat format = FORMAT_HWIO;

  EXPECT_TRUE(FilterFormatFromString("HWIO", &format));
  EXPECT_EQ(format, FORMAT_HWIO);

  format = FORMAT_OIHW;
  EXPECT_TRUE(FilterFormatFromString("DHWIO", &format));
  EXPECT_EQ(format, FORMAT_HWIO);

  format = FORMAT_HWIO;
  EXPECT_TRUE(FilterFormatFromString("OIHW", &format));
  EXPECT_EQ(format, FORMAT_OIHW);

  format = FORMAT_HWIO;
  EXPECT_TRUE(FilterFormatFromString("OIDHW", &format));
  EXPECT_EQ(format, FORMAT_OIHW);
}

TEST(TensorFormatTest, ECP_FilterFormatFromStringVectorFormat) {
  FilterTensorFormat format = FORMAT_HWIO;

  EXPECT_TRUE(FilterFormatFromString("OIHW_VECT_I", &format));
  EXPECT_EQ(format, FORMAT_OIHW_VECT_I);
}

TEST(TensorFormatTest, Invalid_FormatFromStringUnknownReturnsFalse) {
  TensorFormat format = FORMAT_NCHW;

  EXPECT_FALSE(FormatFromString("INVALID", &format));
  EXPECT_EQ(format, FORMAT_NCHW);
}

TEST(TensorFormatTest, Invalid_FilterFormatFromStringUnknownReturnsFalse) {
  FilterTensorFormat format = FORMAT_OIHW;

  EXPECT_FALSE(FilterFormatFromString("INVALID", &format));
  EXPECT_EQ(format, FORMAT_OIHW);
}

TEST(TensorFormatTest, BVA_FormatFromStringEmptyStringReturnsFalse) {
  TensorFormat format = FORMAT_NCHW;

  EXPECT_FALSE(FormatFromString("", &format));
  EXPECT_EQ(format, FORMAT_NCHW);
}

TEST(TensorFormatTest, BVA_FilterFormatFromStringEmptyStringReturnsFalse) {
  FilterTensorFormat format = FORMAT_OIHW;

  EXPECT_FALSE(FilterFormatFromString("", &format));
  EXPECT_EQ(format, FORMAT_OIHW);
}

TEST(TensorFormatTest, Edge_FormatFromStringIsCaseSensitive) {
  TensorFormat format = FORMAT_NCHW;

  EXPECT_FALSE(FormatFromString("nhwc", &format));
  EXPECT_FALSE(FormatFromString("Nhwc", &format));
  EXPECT_FALSE(FormatFromString("nchw", &format));
  EXPECT_EQ(format, FORMAT_NCHW);
}

TEST(TensorFormatTest, Edge_FilterFormatFromStringIsCaseSensitive) {
  FilterTensorFormat format = FORMAT_OIHW;

  EXPECT_FALSE(FilterFormatFromString("hwio", &format));
  EXPECT_FALSE(FilterFormatFromString("Hwio", &format));
  EXPECT_FALSE(FilterFormatFromString("oihw", &format));
  EXPECT_EQ(format, FORMAT_OIHW);
}

TEST(TensorFormatTest, Edge_FormatFromStringDoesNotTrimWhitespace) {
  TensorFormat format = FORMAT_NCHW;

  EXPECT_FALSE(FormatFromString(" NHWC", &format));
  EXPECT_FALSE(FormatFromString("NHWC ", &format));
  EXPECT_FALSE(FormatFromString("\tNHWC", &format));
  EXPECT_FALSE(FormatFromString("NHWC\n", &format));
  EXPECT_EQ(format, FORMAT_NCHW);
}

TEST(TensorFormatTest, Edge_FilterFormatFromStringDoesNotTrimWhitespace) {
  FilterTensorFormat format = FORMAT_OIHW;

  EXPECT_FALSE(FilterFormatFromString(" HWIO", &format));
  EXPECT_FALSE(FilterFormatFromString("HWIO ", &format));
  EXPECT_FALSE(FilterFormatFromString("\tHWIO", &format));
  EXPECT_FALSE(FilterFormatFromString("HWIO\n", &format));
  EXPECT_EQ(format, FORMAT_OIHW);
}

TEST(TensorFormatTest, Edge_FormatFromStringRejectsSimilarNames) {
  TensorFormat format = FORMAT_NCHW;

  EXPECT_FALSE(FormatFromString("NHW", &format));
  EXPECT_FALSE(FormatFromString("NHWCC", &format));
  EXPECT_FALSE(FormatFromString("NHWC_VECT_C", &format));
  EXPECT_FALSE(FormatFromString("NCHW_VECT_W", &format));
  EXPECT_EQ(format, FORMAT_NCHW);
}

TEST(TensorFormatTest, Edge_FilterFormatFromStringRejectsSimilarNames) {
  FilterTensorFormat format = FORMAT_OIHW;

  EXPECT_FALSE(FilterFormatFromString("HWI", &format));
  EXPECT_FALSE(FilterFormatFromString("HWIIO", &format));
  EXPECT_FALSE(FilterFormatFromString("OHWI", &format));
  EXPECT_FALSE(FilterFormatFromString("DHWIO_EXTRA", &format));
  EXPECT_EQ(format, FORMAT_OIHW);
}

TEST(TensorFormatDeathTest, Invalid_ToStringTensorFormatDies) {
#ifndef NDEBUG
  EXPECT_DEATH(ToString(static_cast<TensorFormat>(-1)), "Invalid Format");
#endif
}

TEST(TensorFormatDeathTest, Invalid_ToStringFilterTensorFormatDies) {
#ifndef NDEBUG
  EXPECT_DEATH(ToString(static_cast<FilterTensorFormat>(-1)),
               "Invalid Filter Format");
#endif
}

}  // namespace
}  // namespace tensorflow