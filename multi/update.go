package multi

import "time"

type Updater interface {
	Update(r *Result) Status
}

func doUpdates(r *Result, initialTime time.Time, upd []Updater) Status {
	r.Time = time.Since(initialTime)
	for _, u := range upd {
		st := u.Update(r)
		if st != 0 {
			r.Status = st
		}
	}
	r.Iter++
	return r.Status
}
