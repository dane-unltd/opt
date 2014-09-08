package multi

import (
	"github.com/dane-unltd/opt/uni"
)

type Solver interface {
	Stats() *Stats
}

type Optimizer interface {
	Optimize(obj FdF, in *Solution, u ...Updater) (Status, Solver)
}

//Solve a problem using a gradient based method
func Optimize(obj FdF, in *Solution, p *Params, u ...Updater) (Status, Solver) {
	solver := NewSearchBased(new(LBFGS), uni.NewCubic())
	if p != nil {
		u = append(u, GradConv{p.Accuracy})
		u = append(u, p.Termination)
	}

	status := solver.Optimize(obj, in, u...)

	return status, solver
}
