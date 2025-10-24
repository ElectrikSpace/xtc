/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2024-2026 The XTC Project Authors
 */
#include "perf_event.h"
#include <assert.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <syscall.h>
#include <unistd.h>

#if HAS_PFM
#include <perfmon/pfmlib.h>
#endif

int all_perf_events[PERF_EVENT_NUM] = {
    PERF_EVENT_CYCLES,       PERF_EVENT_CLOCKS,        PERF_EVENT_INSTRS,
    PERF_EVENT_MIGRATIONS,   PERF_EVENT_SWITCHES,      PERF_EVENT_CACHE_ACCESS,
    PERF_EVENT_CACHE_MISSES, PERF_EVENT_BRANCH_INSTRS, PERF_EVENT_BRANCH_MISSES,
};

typedef struct {
  int id;
  int type;
  int num;
  const char *name;
} perf_event_decl_t;

static const perf_event_decl_t perf_events_decl[] = {
    {PERF_EVENT_CYCLES, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "cycles"},
    {PERF_EVENT_CLOCKS, PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_CLOCK, "clocks"},
    {PERF_EVENT_INSTRS, PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS,
     "instructions"},
    {PERF_EVENT_MIGRATIONS, PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS,
     "migrations"},
    {PERF_EVENT_SWITCHES, PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES,
     "context_switches"},
    {PERF_EVENT_CACHE_ACCESS, PERF_TYPE_HARDWARE,
     PERF_COUNT_HW_CACHE_REFERENCES, "cache_access"},
    {PERF_EVENT_CACHE_MISSES, PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES,
     "cache_misses"},
    {PERF_EVENT_BRANCH_INSTRS, PERF_TYPE_HARDWARE,
     PERF_COUNT_HW_BRANCH_INSTRUCTIONS, "branches"},
    {PERF_EVENT_BRANCH_MISSES, PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES,
     "branches_misses"},
};

typedef struct {
  struct perf_event_attr *attr;
  char **fstr;
  size_t size;
  int idx;
  int cpu;
  int flags;
} local_pfm_perf_encode_arg_t;

static int sys_perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                               int cpu, int group_fd, unsigned long flags) {
  long fd;
  fd = syscall(SYS_perf_event_open, hw_event, pid, cpu, group_fd, flags);
  if (fd < 0) {
    perror("sys_perf_event_open failure");
    exit(EXIT_FAILURE);
  }
  assert(fd >= 0);
  return (int)fd;
}

static void init_perf_event_attr(struct perf_event_attr *attr_ptr)
{
  memset(attr_ptr, 0, sizeof(*attr_ptr));
  attr_ptr->size = sizeof(*attr_ptr);
  attr_ptr->exclude_kernel = 1;
  attr_ptr->exclude_hv = 1;
  attr_ptr->exclude_idle = 1;
  attr_ptr->inherit = 1;
  attr_ptr->disabled = 0;
}


int open_perf_event(perf_event_args_t event) {
  if (event.mode == PERF_ARG_GENERIC) {
    struct perf_event_attr attr;
    init_perf_event_attr(&attr);
    return sys_perf_event_open(&attr, 0 /*pid*/, -1 /*cpu*/, -1 /*group_fd*/,
                               0 /*flags*/);
  } else {
    local_pfm_perf_encode_arg_t * perf_gen =   (local_pfm_perf_encode_arg_t *)event.args.config_ptr;
    return sys_perf_event_open(perf_gen->attr,
                               0 /*pid*/, perf_gen->cpu /*cpu*/, -1 /*group_fd*/,
                               perf_gen->flags /*flags*/);
  }
}

static __attribute__((constructor)) void perf_event_init(void) {
#if HAS_PFM
  int res = pfm_initialize();
  if (res != PFM_SUCCESS) {
    fprintf(stderr, "ERROR: cannot initialize libpfm: pfm_initialize(): %s\n",
            pfm_strerror(res));
    exit(EXIT_FAILURE);
  }
#endif /* HAS_PFM */
}

static __attribute__((destructor)) void perf_event_fini(void) {
#if HAS_PFM
  pfm_terminate();
#endif /* HAS_PFM */
}

uint64_t read_perf_event(int perf_fd) {
  uint64_t value;
  ssize_t n;
  n = read(perf_fd, &value, sizeof(value));
  assert(n == sizeof(value));
  return value;
}

void close_perf_event(int perf_fd) { close(perf_fd); }

void open_perf_events(int n_events, const perf_event_args_t *events, int *fds) {
  assert(n_events <= PERF_EVENT_MAX_EVENTS);
  for (int i = 0; i < n_events; i++) {
    fds[i] = open_perf_event(events[i]);
    assert(fds[i] != -1);
  }
}

void close_perf_events(int n_events, const int *fds) {
  for (int i = 0; i < n_events; i++) {
    if (fds[i] >= 0) {
      close_perf_event(fds[i]);
    }
  }
}

uint64_t _tmp_results[PERF_EVENT_MAX_EVENTS];

void reset_perf_events(int n_events, const int *fds, uint64_t *results) {
  for (int i = 0; i < n_events; i++) {
    results[i] = 0;
  }
}

void start_perf_events(int n_events, const int *fds, uint64_t *results) {
  for (int i = 0; i < n_events; i++) {
    if (fds[i] >= 0) {
      _tmp_results[i] = read_perf_event(fds[i]);
    }
  }
}

void stop_perf_events(int n_events, const int *fds, uint64_t *results) {
  for (int i = 0; i < n_events; i++) {
    if (fds[i] >= 0) {
      _tmp_results[i] = read_perf_event(fds[i]) - _tmp_results[i];
      results[i] += _tmp_results[i];
    }
  }
}

int get_perf_event_config(const char *name, perf_event_args_t *event) {
  for (int e = 0; e < sizeof(perf_events_decl) / sizeof(*perf_events_decl);
       e++) {
    if (strcmp(name, perf_events_decl[e].name) == 0) {
      event->mode = PERF_ARG_GENERIC;
      event->args.config_pair.event = perf_events_decl[e].num;
      event->args.config_pair.type = perf_events_decl[e].type;
      return 0;
    }
  }
  
  #if HAS_PFM
  struct perf_event_attr *attr = malloc(sizeof(struct perf_event_attr));
  init_perf_event_attr(attr);

  local_pfm_perf_encode_arg_t *arg = malloc(sizeof(local_pfm_perf_encode_arg_t));

  memset(arg, 0, sizeof(local_pfm_perf_encode_arg_t));

  arg->size = sizeof(local_pfm_perf_encode_arg_t);
  arg->attr = attr;

  int ret = pfm_get_os_event_encoding(name, PFM_PLM3, PFM_OS_PERF_EVENT, arg);

  if (ret != PFM_SUCCESS) {
    return 1;
  }

  event->mode = PERF_ARG_PTR;
  event->args.config_ptr = (const void *)arg;
  #endif /* HAS_PFM */

  return 0;
}

void perf_event_args_destroy(perf_event_args_t args) {
    if (args.mode == PERF_ARG_PTR && args.args.config_ptr != NULL) {
        local_pfm_perf_encode_arg_t* pfm = (local_pfm_perf_encode_arg_t*) args.args.config_ptr;
        free((void*)pfm->attr);
        free((void*)args.args.config_ptr);
        args.args.config_ptr = NULL; 
    }    
}
