package multi

import (
	"time"
)

type Stats struct {
	Iter      int
	Time      time.Duration
	FunEvals  int
	GradEvals int
}

type Result struct {
	*Solution
	Stats
	Status Status
}
