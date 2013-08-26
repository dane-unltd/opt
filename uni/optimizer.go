package uni

type FOptimizer interface {
	OptimizeF(obj F, in *Solution, upd ...Updater) *Result
}

type FdFOptimizer interface {
	OptimizeFdF(obj FdF, in *Solution, upd ...Updater) *Result
}
