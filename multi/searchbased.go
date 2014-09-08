package multi

import (
	"time"
)

type SearchDirectioner interface {
	SearchDirection(s *Solution, d []float64)
}

type SearchBased struct {
	LastDir []float64

	sd    SearchDirectioner
	ls    LineSearcher
	stats Stats
}

func NewSearchBased(sd SearchDirectioner, ls LineSearcher) *SearchBased {
	return &SearchBased{
		sd: sd,
		ls: ls,
	}
}

func (sb *SearchBased) Stats() *Stats {
	return &sb.stats
}

func (sb *SearchBased) Optimize(o FdF, sol *Solution, upd ...Updater) Status {

	obj := Wrapper{Stats: &sb.stats, Func: o}
	sol.check(obj)

	if sb.LastDir == nil {
		sb.LastDir = make([]float64, len(sol.X))
	}

	if len(upd) == 0 {
		upd = append(upd, GradConv{1e-6})
	}

	initialTime := time.Now()

	s := 1.0 //initial step size

	var status Status
	for ; status == NotTerminated; status = doUpdates(sol, &sb.stats, initialTime, upd) {
		s = 1.0

		sb.ls.Search(obj, sol, sb.LastDir, s)
		sb.sd.SearchDirection(sol, sb.LastDir)
	}
	return status
}
