package multi

import "time"

type Updater interface {
	Update(sol *Solution, stats *Stats) Status
}

func doUpdates(sol *Solution, stats *Stats, initialTime time.Time, upd []Updater) Status {
	stats.Time = time.Since(initialTime)
	var status Status
	for _, u := range upd {
		st := u.Update(sol, stats)
		if st != 0 {
			status = st
		}
	}
	stats.Iter++
	return status
}
