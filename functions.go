package opt

import (
	"github.com/dane-unltd/linalg/mat"
	"math"
)

//objective of the form x'*A*x + b'*x + c
func MakeQuadratic(A *mat.Dense, b mat.Vec, c float64) (fun func(mat.Vec) float64, grad func(mat.Vec, mat.Vec)) {
	m, n := A.Dims()
	At := A.TrView()
	if m != n {
		panic("coeff matrix has to be quadratic")
	}
	temp := mat.NewVec(m)
	fun = func(x mat.Vec) float64 {
		val := 0.0
		temp.Apply(A, x)
		val += mat.Dot(x, temp)
		val += mat.Dot(x, b)
		val += c
		return val
	}
	grad = func(x mat.Vec, g mat.Vec) {
		temp.Apply(At, x)
		g.Apply(A, x)
		g.Add(g, temp)
		g.Add(g, b)
	}
	return
}

func MakeRosenbrock() (fun func(mat.Vec) float64, grad func(mat.Vec, mat.Vec)) {
	fun = func(x mat.Vec) float64 {
		sum := 0.0
		for i := 0; i < len(x)-1; i++ {
			sum += math.Pow(1-x[i], 2) +
				100*math.Pow(x[i+1]-math.Pow(x[i], 2), 2)
		}
		return sum
	}
	grad = func(x, g mat.Vec) {
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
