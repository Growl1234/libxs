ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
PROJECT := libxs
DEPDIR = ../..
SRCDIR = .
INCDIR = .
BLDDIR = obj
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)

BLAS = 0
OMP = 1
SYM = 1

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

# header-only when library is not (yet) built
ifeq (,$(XLIB))
ifneq (,$(wildcard $(DEPDIR)/libxs/libxs_source.h))
  DFLAGS += -DLIBXS_SOURCE
  HEADERONLY = 1
endif
endif

# XGBoost (optional): head-to-head comparison in predict_params/predict_crystal
XGB ?= 1
ifneq (0,$(XGB))
  PKGCFG ?= $(call which,pkg-config)
  XGBOOST_ROOT ?= $(shell echo "$$XGBOOST_ROOT")
  ifneq (,$(wildcard $(XGBOOST_ROOT)/include/xgboost/c_api.h))
    XGB_LDIR := $(if $(wildcard $(XGBOOST_ROOT)/lib64),lib64,lib)
    IFLAGS += -I$(call quote,$(XGBOOST_ROOT)/include)
    DFLAGS += -D__XGBOOST
    XGBLDFLAGS := -Wl,--rpath=$(XGBOOST_ROOT)/$(XGB_LDIR) \
                  -L$(XGBOOST_ROOT)/$(XGB_LDIR) -lxgboost
  else ifneq (,$(shell $(PKGCFG) --exists xgboost 2>/dev/null && echo yes))
    IFLAGS += $(shell $(PKGCFG) --cflags xgboost)
    DFLAGS += -D__XGBOOST
    XGBLDFLAGS := $(shell $(PKGCFG) --libs xgboost)
  endif
endif

# include directories
IFLAGS += -I$(call quote,$(INCDIR))
IFLAGS += -I$(call quote,$(DEPDIR))
IFLAGS += -I$(call quote,$(DEPDIR)/libxs)

HEADERS := $(wildcard $(INCDIR)/*.h) \
           $(wildcard $(SRCDIR)/*.h) \
           $(wildcard $(DEPDIR)/libxs/*.h) \
           $(DEPDIR)/libxs/libxs_source.h
CSOURCS := $(wildcard $(SRCDIR)/*.c)
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
SOURCES := $(CSOURCS)
OBJECTS := $(COBJCTS)
XFILES := $(OUTDIR)/predict_params.x $(OUTDIR)/predict_sunspots.x \
          $(OUTDIR)/predict_earthquakes.x $(OUTDIR)/predict_discharge.x \
          $(OUTDIR)/predict_soi.x $(OUTDIR)/predict_stock.x \
          $(OUTDIR)/predict_crystal.x $(OUTDIR)/predict_ett.x \
          $(OUTDIR)/predict_prob.x $(OUTDIR)/predict_dist.x \
          $(OUTDIR)/predict_bits.x

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS)

$(OUTDIR)/predict_params.x: $(OUTDIR)/.make $(BLDDIR)/predict_params-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_params-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(XGBLDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_sunspots.x: $(OUTDIR)/.make $(BLDDIR)/predict_sunspots-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_sunspots-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_earthquakes.x: $(OUTDIR)/.make $(BLDDIR)/predict_earthquakes-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_earthquakes-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(XGBLDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_discharge.x: $(OUTDIR)/.make $(BLDDIR)/predict_discharge-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_discharge-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_soi.x: $(OUTDIR)/.make $(BLDDIR)/predict_soi-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_soi-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_stock.x: $(OUTDIR)/.make $(BLDDIR)/predict_stock-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_stock-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_crystal.x: $(OUTDIR)/.make $(BLDDIR)/predict_crystal-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_crystal-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(XGBLDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_ett.x: $(OUTDIR)/.make $(BLDDIR)/predict_ett-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_ett-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(XGBLDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_prob.x: $(OUTDIR)/.make $(BLDDIR)/predict_prob-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_prob-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_bits.x: $(OUTDIR)/.make $(BLDDIR)/predict_bits-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_bits-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(OUTDIR)/predict_dist.x: $(OUTDIR)/.make $(BLDDIR)/predict_dist-c.o $(XLIB)
	$(LD) -o $@ $(BLDDIR)/predict_dist-c.o $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)


$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

.PHONY: clean
clean:
ifneq ($(call qapath,$(BLDDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BLDDIR)),$(call qapath,.))
	@-rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@-rm -f $(OBJECTS)
endif

.PHONY: realclean
realclean: clean
ifneq ($(call qapath,$(OUTDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(OUTDIR)),$(call qapath,.))
	@-rm -rf $(OUTDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR))) # still exists
	@-rm -f $(XFILES) $(MODULES) *.bin
endif

.PHONY: deepclean
deepclean: realclean
	@-rm -f .make .state
