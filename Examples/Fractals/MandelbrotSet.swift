//******************************************************************************
// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
import Foundation
import SwiftRT
#if canImport(SwiftRTCuda)
import SwiftRTCuda
#endif

func mandelbrotSet(
  iterations: Int,
  tolerance: Float,
  range: ComplexRange,
  size imageSize: ImageSize,
  mode: FractalCalculationMode
) -> Tensor2 {
  let size = (r: imageSize[0], c: imageSize[1])

  let rFirst = Complex<Float>(range.start.real, 0)
  let rLast  = Complex<Float>(range.end.real, 0)
  let iFirst = Complex<Float>(0, range.start.imaginary)
  let iLast  = Complex<Float>(0, range.end.imaginary)

  // repeat rows of real range, columns of imaginary range, and combine
  let X = repeating(array(from: rFirst, to: rLast, (1, size.c)), size) +
          repeating(array(from: iFirst, to: iLast, (size.r, 1)), size)
  var Z = X
  var divergence = mode == .kernel ? empty(size) : full(size, iterations)

  print("rows: \(size.r), cols: \(size.c), iterations: \(iterations)")
  let start = Date()
  switch mode {
  case .direct:
    for i in 1..<iterations {
      divergence[abs(Z) .> tolerance] = min(divergence, i)
      Z = multiply(Z, Z, add: X)
    }

  case .parallelMap:
    // TODO: Have pmap take in X.
    // pmap(Z, X, &divergence) { Z, X, divergence in
    pmap(Z, &divergence) { Z, divergence in
      for i in 1..<iterations {
        divergence[abs(Z) .> tolerance] = min(divergence, i)
        Z = multiply(Z, Z, add: X)
      }
    }

  case .kernel:
    #if canImport(SwiftRTCuda)
      let queue = currentQueue

      srtMandelbrotFlat(
          Complex<Float>.type,
          X.deviceRead(using: queue),
          tolerance,
          iterations,
          divergence.count,
          divergence.deviceReadWrite(using: queue),
          queue.stream)

        queue.waitForCompletion()
    #else
    // TODO: Have kernel take in X.
    pmap(Z, &divergence, limitedBy: .compute) {
      mandelbrotKernel(Z: $0, divergence: &$1, tolerance, iterations)
    }
    #endif
  }
  
  print("MandelbrotSet elapsed \(String(format: "%.7f", Date().timeIntervalSince(start))) seconds")
  return divergence
}

@inlinable public func mandelbrotKernel<E>(
  Z: TensorR2<Complex<E>>,
  divergence: inout TensorR2<E>,
  _ tolerance: E,
  _ iterations: Int
) where E == E.Value, E.Value: Real & Comparable {
  let message =
    "mandelbrot(Z: \(Z.name), divergence: \(divergence.name), " +
    "tolerance: \(tolerance), iterations: \(iterations))"

  kernel(Z, &divergence, message) {
    let x = $0; var z = x, d = $1, i = E.Value.zero

    while abs(z) <= tolerance && i < d {
        z = z * z + x
        i += 1
    }
    return i
  }
}
