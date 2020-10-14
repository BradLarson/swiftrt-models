# SwiftRT Models

This repository contains models and examples written in Swift that use [SwiftRT](https://github.com/ewconnell/swiftrt).

## Building with Swift Package Manager (CPU only)

Swift Package Manager can be used to build CPU-only versions of the examples and tests. For CUDA capabilities, you'll need to build with CMake (see below). To build all targets within the project, use:

```
swift build
```

Unit tests can be run with

```
swift test
```

To run a specific example via Swift Package Manager, you can use the following:

```
swift run -c release Fractals [arguments]
```

## Building with CMake (CPU and CUDA)

CMake can build CUDA-capable models and examples, in addition to ones that run on the CPU. Currently, CMake 3.18 is required to build SwiftRT and these models. As an example for how you can build using CMake, change to the project directory and run:

```
cmake -B build -D SWIFTRT_ENABLE_CUDA=YES -D CMAKE_CUDA_ARCHITECTURES="60;70;75" -D BUILD_TESTING=YES -D CMAKE_BUILD_TYPE=Release -D CMAKE_Swift_COMPILER=$(which swiftc) -G Ninja -S ./
cmake --build ./build
```

This will use the ./build directory to store the build configuration files and perform the build. Binaries can then be found within ./build/bin.

## Bugs

Please report model-related bugs and feature requests using GitHub issues in
this repository.

## Community

Discussion about Swift for TensorFlow happens on the
[swift@tensorflow.org](https://groups.google.com/a/tensorflow.org/d/forum/swift)
mailing list.

## Contributing

We welcome contributions: please read the [Contributor Guide](CONTRIBUTING.md)
to get started. It's always a good idea to discuss your plans on the mailing
list before making any major submissions.

## Code of Conduct

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of
experience, education, socio-economic status, nationality, personal appearance,
race, religion, or sexual identity and orientation.

The Swift for TensorFlow community is guided by our [Code of
Conduct](CODE_OF_CONDUCT.md), which we encourage everybody to read before
participating.
