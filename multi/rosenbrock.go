package multi

import (
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt/uni"
	"math"
)

type Rosenbrock struct {
	LineSearch uni.Solver
}

func NewRosenbrock() *Rosenbrock {
	return &Rosenbrock{
		LineSearch: uni.NewQuadratic(),
	}
}

func (sol *Rosenbrock) Solve(m *Model) {
	eps := 1.0

	m.init(false, false)

	d := make([]mat.Vec, m.N)
	for i := range d {
		d[i] = mat.NewVec(m.N)
		d[i][i] = 1
	}

	lambda := make([]float64, m.N)

	lf := make([]*LineFunc, m.N)
	for i := range lf {
		lf[i] = NewLineFunc(m.Obj, m.X, d[i])
	}

	lsInit := uni.NewSolution()
	lsParams := uni.NewParams()
	lsParams.XTolAbs = m.Params.XTolAbs
	lsParams.XTolRel = m.Params.XTolRel
	lsParams.FunTolAbs = 0
	lsParams.FunTolRel = 0

	for {
		if m.Status = m.update(); m.Status != 0 {
			return
		}

		//Search in all directions
		for i := range d {
			lf[i].Dir = 1
			valNeg := 0.0
			valPos := lf[i].Val(eps)
			m.FunEvals++
			if valPos >= m.ObjX {
				lf[i].Dir = -1
				valNeg = lf[i].Val(eps)
				m.FunEvals++
				if valNeg >= m.ObjX {
					eps *= 0.5
					lf[i].Dir = 1
					lsInit.SetLB(-eps)
					lsInit.SetUB(eps)
					lsInit.SetX(0)
				} else {
					lsInit.SetUB()
					lsInit.SetLB()
					lsInit.SetX(eps)
				}
			} else {
				lsInit.SetUB()
				lsInit.SetLB()
				lsInit.SetX(eps)
			}
			lsRes := sol.LineSearch.Solve(lf[i], lsInit, lsParams)
			m.FunEvals += lsRes.FunEvals

			lambda[i] = lf[i].Dir * lsRes.X
			m.X.Axpy(lambda[i], d[i])
			m.ObjX = lsRes.ObjX
		}

		//Find new directions
		for i := range d {
			if math.Abs(lambda[i]) > m.Params.XTolAbs {
				d[i].Scal(lambda[i])
				for j := i + 1; j < m.N; j++ {
					d[i].Axpy(lambda[j], d[j])
				}
			}
		}

		//Gram-Schmidt, TODO:use QR factorization
		for i := range d {
			d[i].Scal(1 / d[i].Nrm2())
			for j := i + 1; j < m.N; j++ {
				d[j].Axpy(-mat.Dot(d[i], d[j]), d[i])
			}
		}

	}
}
