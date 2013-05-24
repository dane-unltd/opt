package unc

import (
	"github.com/dane-unltd/linalg/matrix"
	"math"
)

//objective of the form x'*A*x + b'*x + c
func MakeQuadratic(A *matrix.Dense, b matrix.Vec, c float64) (fun Miso, grad Mimo) {
	m, n := A.Size()
	At := A.TrView()
	if m != n {
		panic("coeff matrix has to be quadratic")
	}
	temp := matrix.NewVec(m)
	fun = func(x matrix.Vec) float64 {
		val := 0.0
		temp.Mul(A, x)
		val += matrix.Dot(x, temp)
		val += matrix.Dot(x, b)
		val += c
		return val
	}
	grad = func(x matrix.Vec, g matrix.Vec) {
		temp.Mul(At, x)
		g.Mul(A, x)
		g.Add(g, temp)
		g.Add(g, b)
	}
	return
}

func MakeRosenbrock() (fun Miso, grad Mimo) {
	fun = func(x matrix.Vec) float64 {
		sum := 0.0
		for i := 0; i < len(x)-1; i++ {
			sum += math.Pow(1-x[i], 2) +
				100*math.Pow(x[i+1]-math.Pow(x[i], 2), 2)
		}
		return sum
	}
	grad = func(x, g matrix.Vec) {
		g[len(x)-1] = 0
		for i := 0; i < len(x)-1; i++ {
			g[i] = -1 * 2 * (1 - x[i])
			g[i] += 2 * 100 * (x[i+1] - math.Pow(x[i], 2)) * (-2 * x[i])
		}
		for i := 1; i < len(x); i++ {
			g[i] += 2 * 100 * (x[i] - math.Pow(x[i-1], 2))
		}
	}
	return
}
