package multi

import (
	"github.com/dane-unltd/opt/uni"
	"github.com/dane-unltd/goblas"
	"math"
	"time"
)

type Rosenbrock struct {
	Termination
	LineSearch uni.FOptimizer
	Accuracy   float64
}

func NewRosenbrock() *Rosenbrock {
	return &Rosenbrock{
		Termination: Termination{
			IterMax: 1000,
			TimeMax: time.Minute,
		},
		LineSearch: uni.NewQuadratic(),
		Accuracy:   1e-4,
	}
}
func (sol *Rosenbrock) OptimizeFdF(o FdF, in *Solution, upd ...Updater) *Result {
	return sol.OptimizeF(o, in, upd...)
}

func (sol *Rosenbrock) OptimizeF(o F, in *Solution, upd ...Updater) *Result {
	r := NewResult(in)
	obj := fWrapper{r: r, f: o}
	r.initF(obj)

	upd = append(upd, sol.Termination)

	initialTime := time.Now()

	eps := 1.0
	n := len(r.X)
	x := goblas.NewVector(r.X)

	d := make([]goblas.Vector, n)
	for i := range d {
		d[i] = goblas.NewVector(make([]float64, n))
		d[i].Data[i] = 1
	}

	lambda := make([]float64, n)

	lf := make([]*LineF, n)
	for i := range lf {
		lf[i] = NewLineF(obj, r.X, d[i].Data)
	}

	lsInit := uni.NewSolution()

	for doUpdates(r, initialTime, upd) == 0 {

		//Search in all directions
		for i := range d {
			lf[i].Dir = 1
			valNeg := 0.0
			valPos := lf[i].F(eps)
			if valPos >= r.Obj {
				lf[i].Dir = -1
				valNeg = lf[i].F(eps)
				if valNeg >= r.Obj {
					lf[i].Dir = 1
					lsInit.SetLower(-eps, valNeg)
					lsInit.SetUpper(eps, valPos)
					lsInit.Set(0, r.Obj)
					eps *= 0.5
				} else {
					lsInit.SetUpper()
					lsInit.SetLower()
					lsInit.Set(eps, valNeg)
				}
			} else {
				lsInit.SetUpper()
				lsInit.SetLower()
				lsInit.Set(eps, valPos)
			}
			lsRes := sol.LineSearch.OptimizeF(lf[i], lsInit,
				uni.Accuracy(eps*0.5))

			lambda[i] = lf[i].Dir * lsRes.X

			goblas.Daxpy(lambda[i], d[i], x)
			r.Obj = lsRes.Obj
		}

		//Find new directions
		for i := range d {
			if math.Abs(lambda[i]) > sol.Accuracy {
				goblas.Dscal(lambda[i], d[i])
				for j := i + 1; j < n; j++ {
					goblas.Daxpy(lambda[j], d[j], d[i])
				}
			}
		}

		//Gram-Schmidt, TODO:use QR factorization
		for i := range d {
			goblas.Dscal(1/goblas.Dnrm2(d[i]), d[i])
			for j := i + 1; j < n; j++ {
				goblas.Daxpy(-goblas.Ddot(d[i], d[j]), d[i], d[j])
			}
		}

	}
	return r
}
