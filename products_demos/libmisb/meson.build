project('libmisb', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++17'])

libmisb_dep = dependency('misb-0.0', method : 'pkg-config')


executable('libmisb_app',
           'libmisb_app.cpp',
           install : true,
           dependencies: [libmisb_dep])
