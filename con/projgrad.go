package con

import (
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt"
	"github.com/dane-unltd/opt/linesearch"
)

type ProjGradSolver struct {
	Tol        float64
	IterMax    int
	LineSearch linesearch.Solver
}

func (sol ProjGradSolver) Solve(obj opt.Miso, grad opt.Mimo, proj opt.Projection, x mat.Vec) opt.Result {
	s := 1.0
	fHist := make([]float64, 0)
	proj(x)
	f := obj(x)
	fHist = append(fHist, f)
	d := mat.NewVec(len(x))
	gLin := 0.0

	xTemp := mat.NewVec(len(x))

	lineFun := func(s float64) float64 {
		xTemp.Copy(x)
		xTemp.Axpy(s, d)
		proj(xTemp)
		return obj(xTemp)
	}
	i := 0
	for ; i < sol.IterMax; i++ {
		grad(x, d)
		d.Scal(-1)

		xTemp.Copy(x)
		xTemp.Axpy(s/2, d)
		proj(xTemp)
		xTemp.Sub(xTemp, x)
		xTemp.Scal(2 / s)

		gLin = -xTemp.Nrm2Sq()

		if gLin/float64(len(x)) > -sol.Tol {
			break
		}

		s, f = sol.LineSearch.Solve(lineFun, nil, f, gLin, s)
		fHist = append(fHist, f)

		x.Axpy(s, d)
		proj(x)
	}
	res := opt.Result{
		Obj:     f,
		Iter:    i,
		Grad:    d.Scal(-1),
		ObjHist: fHist,
	}
	if i == sol.IterMax {
		res.Status = opt.MaxIter
	}
	return res
}
