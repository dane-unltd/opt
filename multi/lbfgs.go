package multi

import (
	"fmt"
	"github.com/dane-unltd/opt/uni"
	"github.com/gonum/blas/blasw"
	"time"
)

type LBFGS struct {
	Termination
	Mem        int
	LineSearch uni.FdFOptimizer
}

func NewLBFGS() *LBFGS {
	s := &LBFGS{
		Termination: Termination{
			IterMax: 1000,
			TimeMax: time.Minute,
		},
		Mem:        5,
		LineSearch: uni.NewCubic(),
	}
	return s
}

func (sol LBFGS) OptimizeFdF(o FdF, in *Solution, upd ...Updater) *Result {
	r := NewResult(in)
	obj := fdfWrapper{r: r, fdf: o}
	r.initFdF(obj)

	upd = append(upd, sol.Termination)

	initialTime := time.Now()

	gLin := 0.0
	n := len(r.X)

	S := make([]blasw.Vector, sol.Mem)
	Y := make([]blasw.Vector, sol.Mem)
	for i := 0; i < sol.Mem; i++ {
		S[i] = blasw.NewVector(make([]float64, n))
		Y[i] = blasw.NewVector(make([]float64, n))
	}

	x := blasw.NewVector(r.X)
	g := blasw.NewVector(r.Grad)
	d := blasw.NewVector(make([]float64, n))

	xOld := blasw.NewVector(make([]float64, n))
	gOld := blasw.NewVector(make([]float64, n))
	sNew := blasw.NewVector(make([]float64, n))
	yNew := blasw.NewVector(make([]float64, n))

	alphas := make([]float64, sol.Mem)
	betas := make([]float64, sol.Mem)
	rhos := make([]float64, sol.Mem)

	lineFunc := NewLineFdF(obj, r.X, d.Data)
	lsInit := uni.NewSolution()

	notFirst := false
	for doUpdates(r, initialTime, upd) == 0 {
		blasw.Copy(g, d)
		if notFirst {
			blasw.Copy(g, yNew)
			blasw.Axpy(-1, gOld, yNew)
			blasw.Copy(x, sNew)
			blasw.Axpy(-1, xOld, sNew)

			temp := S[len(S)-1]
			copy(S[1:], S)
			S[0] = temp
			blasw.Copy(sNew, S[0])

			temp = Y[len(S)-1]
			copy(Y[1:], Y)
			Y[0] = temp
			blasw.Copy(yNew, Y[0])

			copy(rhos[1:], rhos)
			rhos[0] = 1 / blasw.Dot(sNew, yNew)
			for i := 0; i < sol.Mem; i++ {
				alphas[i] = rhos[i] * blasw.Dot(S[i], d)
				blasw.Axpy(-alphas[i], Y[i], d)
			}
			for i := sol.Mem - 1; i >= 0; i-- {
				betas[i] = rhos[i] * blasw.Dot(Y[i], d)
				blasw.Axpy(alphas[i]-betas[i], S[i], d)
			}
		}
		notFirst = true

		blasw.Scal(-1, d)
		gLin = blasw.Dot(d, g)

		wolfe := uni.Wolfe{
			Armijo:    0.2,
			Curvature: 0.9,
			X0:        0,
			F0:        r.Obj,
			Deriv0:    gLin,
		}

		lsInit.Set(1.0)
		lsInit.SetLower(0, r.Obj, gLin)
		lsRes := sol.LineSearch.OptimizeFdF(lineFunc, lsInit, wolfe)
		if lsRes.Status < 0 {
			fmt.Println("Linesearch:", lsRes.Status)
			blasw.Copy(g, d)
			blasw.Scal(-1, d)
			lsInit.SetLower(0, r.Obj, blasw.Nrm2(g))
			lsRes = sol.LineSearch.OptimizeFdF(lineFunc, lsInit, wolfe)
			if lsRes.Status < 0 {
				fmt.Println("Linesearch:", lsRes.Status)
				r.Status = Status(lsRes.Status)

				break
			}
		}
		stepSize := lsRes.X
		r.Obj = lsRes.Obj

		blasw.Copy(x, xOld)
		blasw.Copy(g, gOld)

		blasw.Axpy(stepSize, d, x)
		obj.DF(r.X, r.Grad)
	}
	return r
}
