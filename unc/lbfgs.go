package unc

import (
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt"
	"github.com/dane-unltd/opt/linesearch"
)

type LBFGSSolver struct {
	Tol        float64
	IterMax    int
	Mem        int
	LineSearch linesearch.Solver
}

func (sol LBFGSSolver) Solve(obj opt.Miso, grad opt.Mimo, x mat.Vec) opt.Result {
	stepSize := 1.0
	n := len(x)

	fHist := make([]float64, 0)
	f := obj(x)
	fHist = append(fHist, f)
	gLin := 0.0

	S := make([]mat.Vec, sol.Mem)
	Y := make([]mat.Vec, sol.Mem)
	for i := 0; i < sol.Mem; i++ {
		S[i] = mat.NewVec(n)
		Y[i] = mat.NewVec(n)
	}

	d := mat.NewVec(n)
	g := mat.NewVec(n)

	xOld := mat.NewVec(n)
	gOld := mat.NewVec(n)
	sNew := mat.NewVec(n)
	yNew := mat.NewVec(n)

	alphas := mat.NewVec(sol.Mem)
	betas := mat.NewVec(sol.Mem)
	rhos := mat.NewVec(sol.Mem)

	xTemp := mat.NewVec(n)
	lineFun := func(step float64) float64 {
		xTemp.Copy(x)
		xTemp.Axpy(step, d)
		return obj(xTemp)
	}

	iter := 0
	for ; iter < sol.IterMax; iter++ {
		grad(x, g)
		d.Copy(g)

		if iter > 0 {
			yNew.Sub(g, gOld)
			sNew.Sub(x, xOld)

			temp := S[len(S)-1]
			copy(S[1:], S)
			S[0] = temp
			S[0].Copy(sNew)

			temp = Y[len(S)-1]
			copy(Y[1:], Y)
			Y[0] = temp
			Y[0].Copy(yNew)

			copy(rhos[1:], rhos)
			rhos[0] = 1 / mat.Dot(sNew, yNew)
			for i := 0; i < sol.Mem; i++ {
				alphas[i] = rhos[i] * mat.Dot(S[i], d)
				d.Axpy(-alphas[i], Y[i])
			}
			for i := sol.Mem - 1; i >= 0; i-- {
				betas[i] = rhos[i] * mat.Dot(Y[i], d)
				d.Axpy(alphas[i]-betas[i], S[i])
			}
		}

		d.Scal(-1)

		gLin = mat.Dot(d, g)

		if gLin/float64(len(x)) > -sol.Tol {
			break
		}

		stepSize, f = sol.LineSearch.Solve(lineFun, nil, f, gLin,
			stepSize)

		xOld.Copy(x)
		gOld.Copy(g)

		fHist = append(fHist, f)

		x.Axpy(stepSize, d)
	}
	res := opt.Result{
		Obj:     f,
		Iter:    iter,
		Grad:    g,
		ObjHist: fHist,
	}
	if iter == sol.IterMax {
		res.Status = opt.MaxIter
	}
	return res
}
