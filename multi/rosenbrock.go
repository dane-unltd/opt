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

func (sol *Rosenbrock) Solve(o Function, in *Solution, p *Params, u ...Updater) *Result {
	r := NewResult(in)
	obj := ObjWrapper{r: r, o: o}
	r.init(obj)
	h := NewHelper(r.Solution, u)

	eps := 1.0
	n := len(r.X)

	d := make([]mat.Vec, n)
	for i := range d {
		d[i] = mat.NewVec(n)
		d[i][i] = 1
	}

	lambda := make([]float64, n)

	lf := make([]*LineFunc, n)
	for i := range lf {
		lf[i] = NewLineFunc(obj, r.X, d[i])
	}

	lsInit := uni.NewSolution()
	lsParams := uni.NewParams()
	lsParams.XTolAbs = p.XTolAbs
	lsParams.XTolRel = p.XTolRel
	lsParams.FunTolAbs = 0
	lsParams.FunTolRel = 0

	for ; r.Status == NotTerminated; h.update(r, p) {

		//Search in all directions
		for i := range d {
			lf[i].Dir = 1
			valNeg := 0.0
			valPos := lf[i].Val(eps)
			if valPos >= r.ObjX {
				lf[i].Dir = -1
				valNeg = lf[i].Val(eps)
				if valNeg >= r.ObjX {
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

			lambda[i] = lf[i].Dir * lsRes.X
			r.X.Axpy(lambda[i], d[i])
			r.ObjX = lsRes.ObjX
		}

		//Find new directions
		for i := range d {
			if math.Abs(lambda[i]) > p.XTolAbs {
				d[i].Scal(lambda[i])
				for j := i + 1; j < n; j++ {
					d[i].Axpy(lambda[j], d[j])
				}
			}
		}

		//Gram-Schmidt, TODO:use QR factorization
		for i := range d {
			d[i].Scal(1 / d[i].Nrm2())
			for j := i + 1; j < n; j++ {
				d[j].Axpy(-mat.Dot(d[i], d[j]), d[i])
			}
		}

	}
	return r
}
