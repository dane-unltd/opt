package multi

import (
	"github.com/dane-unltd/opt/uni"
)

type FOptimixer interface {
	OptimizeF(obj F, in *Solution, u ...Updater) *Result
}

type FdFOptimizer interface {
	OptimizeFdF(obj FdF, in *Solution, u ...Updater) *Result
}

type FProjOptimizer interface {
	OptimizeFProj(obj FdF, proj Projection, in *Solution, u ...Updater) *Result
}

//Solve a problem choosing an appropriate solver.
func OptimizeF(obj F, in *Solution, p *Params, u ...Updater) *Result {
	if obj, ok := obj.(FdF); ok {
		return OptimizeFdF(obj, in, p, u...)
	}

	solver := NewRosenbrock()
	if p != nil {
		solver.Termination = p.Termination
		solver.Accuracy = p.Accuracy
		u = append(u, NewDeltaXConv(p.Accuracy))
	}
	return solver.OptimizeF(obj, in, u...)
}

//Solve a problem using a gradient based method
func OptimizeFdF(obj FdF, in *Solution, p *Params, u ...Updater) *Result {
	solver := NewSearchBased(new(LBFGS), uni.NewCubic())
	if p != nil {
		u = append(u, GradConv{p.Accuracy})
		u = append(u, p.Termination)
	}

	return solver.OptimizeFdF(obj, in, u...)
}

//Solve a constrained problem using a gradient and projection based methodbw.
func OptimizeFProj(obj FdF, pr Projection, in *Solution, p *Params, u ...Updater) *Result {
	solver := NewProjGrad()
	if p != nil {
		solver.Termination = p.Termination
		u = append(u, NewDeltaXConv(p.Accuracy))
	}

	return solver.OptimizeFProj(obj, pr, in, u...)
}
