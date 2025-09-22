#include <cxxabi.h> // For demangling
#include <iostream>
#include <typeinfo>

template <typename T> void print_type(const T &obj) {
  int status;
  const char *name = typeid(obj).name();
  char *demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
  std::cout << (status == 0 ? demangled : name) << std::endl;
  free(demangled);
}
