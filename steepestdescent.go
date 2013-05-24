package opt

import (
	"github.com/dane-unltd/linalg/matrix"
	"github.com/dane-unltd/opt/linesearch"
)

type SteepestDescentSolver struct {
	Tol        float64
	IterMax    int
	LineSearch linesearch.Solver
}

func (sd SteepestDescentSolver) Solve(obj Miso, grad Mimo, x matrix.Vec) Result {
	s := 1.0
	f := obj(x)
	d := matrix.NewVec(len(x))
	gLin := 0.0

	xTemp := matrix.NewVec(len(x))

	lineFun := func(s float64) float64 {
		xTemp.Copy(x)
		xTemp.Axpy(s, d)
		return obj(xTemp)
	}
	i := 0
	for ; i < sd.IterMax; i++ {
		grad(x, d)
		d.Scal(-1)

		gLin = -d.Nrm2Sq()

		if gLin/float64(len(x)) > -sd.Tol {
			break
		}

		s, f = sd.LineSearch.Solve(lineFun, nil, f, gLin, s)

		x.Axpy(s, d)
	}
	res := Result{
		Obj:  f,
		Iter: i,
		Grad: d.Scal(-1),
	}
	if i == sd.IterMax {
		res.Status = MaxIter
	}
	return res
}
