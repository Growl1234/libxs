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
SSE = 0

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

# header-only when library is not (yet) built
ifeq (,$(XLIB))
ifneq (,$(wildcard $(DEPDIR)/libxs/libxs_source.h))
  DFLAGS += -DLIBXS_SOURCE
  HEADERONLY = 1
endif
endif

# include directories
IFLAGS += -I$(call quote,$(INCDIR))
IFLAGS += -I$(call quote,$(DEPDIR))
IFLAGS += -I$(call quote,$(DEPDIR)/libxs)

OUTNAME := $(shell basename "$(ROOTDIR)")
HEADERS := $(wildcard $(INCDIR)/*.h) \
           $(wildcard $(SRCDIR)/*.h) \
           $(wildcard $(DEPDIR)/libxs/*.h) \
           $(DEPDIR)/libxs/libxs_source.h
CSOURCS := $(wildcard $(SRCDIR)/*.c)
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
# Fortran variant (vpack.f) unless no Fortran compiler is found or FORTRAN=0
ifneq (,$(strip $(FC)))
  F77SRCS := $(wildcard $(SRCDIR)/*.f)
  F77OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F77SRCS:.f=-f77.o)))
  DFLAGS += -DPACK_FORTRAN
  # the main program is C
  ifneq (0,$(INTEL))
  ifeq (,$(strip $(GFC)))
    FLDMAIN := -nofor-main
  endif
  endif
endif
SOURCES := $(CSOURCS) $(F77SRCS)
OBJECTS := $(COBJCTS)
FTNOBJS := $(F77OBJS)
XFILES := $(OUTDIR)/$(OUTNAME).x

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS) $(FTNOBJS)

ifneq (,$(strip $(FTNOBJS)))
$(OUTDIR)/$(OUTNAME).x: $(OUTDIR)/.make $(OBJECTS) $(FTNOBJS) $(XLIB)
	$(FLD) -o $@ $(OBJECTS) $(FTNOBJS) $(XLIB) $(FLDMAIN) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
else
$(OUTDIR)/$(OUTNAME).x: $(OUTDIR)/.make $(OBJECTS) $(XLIB)
	$(LD) -o $@ $(OBJECTS) $(XLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
endif

$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CXX) $(DFLAGS) $(IFLAGS) $(CXXFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-f77.o: $(SRCDIR)/%.f .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@ $(FMFLAGS) $(BLDDIR)

.PHONY: clean
clean:
ifneq ($(call qapath,$(BLDDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BLDDIR)),$(call qapath,.))
	@-rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@-rm -f $(OBJECTS) $(FTNOBJS)
endif

.PHONY: realclean
realclean: clean
ifneq ($(call qapath,$(OUTDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(OUTDIR)),$(call qapath,.))
	@-rm -rf $(OUTDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR))) # still exists
	@-rm -f $(XFILES) $(MODULES)
endif

.PHONY: deepclean
deepclean: realclean
	@-rm -f .make .state

# Relink when only the link flags changed: those are tracked apart from .state so
# that a link-only change no longer recompiles every object, which leaves the link
# as the one step that still has to see them.
$(XFILES): $(LNKSTATE)
