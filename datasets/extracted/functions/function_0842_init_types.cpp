#include "mlir/IR/BuiltinTypes.h"  
#include "tensorflow/compiler/mlir/python/mlir_wrapper/mlir_wrapper.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
void init_types(py::module& m) {
  py::class_<mlir::Type> Type(m, "Type");
  py::class_<mlir::FunctionType, mlir::Type>(m, "FunctionType")
      .def("getResults",
           [](mlir::FunctionType& ft) { return ft.getResults().vec(); });
  py::class_<mlir::FloatType, mlir::Type>(m, "FloatType")
      .def("getBF16", &mlir::FloatType::getBF16)
      .def("getF16", &mlir::FloatType::getF16)
      .def("getF32", &mlir::FloatType::getF32)
      .def("getF64", &mlir::FloatType::getF64);
  py::class_<mlir::IntegerType, mlir::Type>(m, "IntegerType")
      .def("get", [](mlir::MLIRContext* context, unsigned width) {
        return mlir::IntegerType::get(context, width,
                                      mlir::IntegerType::Signless);
      });
  py::class_<mlir::UnrankedTensorType, mlir::Type>(m, "UnrankedTensorType")
      .def("get", &mlir::UnrankedTensorType::get);
  py::class_<mlir::RankedTensorType, mlir::Type>(m, "RankedTensorType")
      .def("get", [](std::vector<int64_t> shape, mlir::Type ty) {
        return mlir::RankedTensorType::get(mlir::ArrayRef<int64_t>(shape), ty);
      });
}