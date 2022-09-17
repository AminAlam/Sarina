#!/bin/bash
cpp_fiel_path='src/lib/'
g++ -c -fPIC $cpp_fiel_path'cpp_backend.cpp' -o $cpp_fiel_path'cpp_backend.o'
g++ -shared -Wl -o $cpp_fiel_path'lib_cpp_backend.so'  $cpp_fiel_path'cpp_backend.o' -v