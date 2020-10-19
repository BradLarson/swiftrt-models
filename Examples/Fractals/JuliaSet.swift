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

func juliaSet(
    iterations: Int,
    constant C: Complex<Float>,
    tolerance: Float,
    range: ComplexRange,
    size: ImageSize
) -> Tensor2 {
    let size2 = (r: size[0], c: size[1])

    let rFirst = Complex<Float>(range.start.real, 0), rLast = Complex<Float>(range.end.real, 0)
    let iFirst = Complex<Float>(0, range.start.imaginary), iLast = Complex<Float>(0, range.end.imaginary)

    let Xr = repeating(array(from: rFirst, to: rLast, (1, size2.c)), size2)
    let Xi = repeating(array(from: iFirst, to: iLast, (size2.r, 1)), size2)
    var Z = Xr + Xi

    var divergence = full(size, iterations)

    measureTime {
        for i in 0..<iterations {
            Z = multiply(Z, Z, add: C)
            divergence[abs(Z) .> tolerance] = min(divergence, i)
        }
    }
    return divergence
}
