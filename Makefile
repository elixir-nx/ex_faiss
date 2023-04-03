# Environment variables passed via elixir_make
# ERTS_INCLUDE_DIR
# MIX_APP_PATH

TEMP ?= $(HOME)/.cache
FAISS_CACHE ?= $(TEMP)/ex_faiss
FAISS_GIT_REPO ?= https://www.github.com/facebookresearch/faiss
FAISS_GIT_REV ?= 19f7696deedc93615c3ee0ff4de22284b53e0243
FAISS_NS = faiss-$(FAISS_GIT_REV)
FAISS_DIR = $(FAISS_CACHE)/$(FAISS_NS)
FAISS_LIB_DIR = $(FAISS_DIR)/build/faiss
FAISS_LIB_DIR_FLAG = $(FAISS_DIR)/build/faiss/ex_faiss.ok

# Private configuration
PRIV_DIR = $(MIX_APP_PATH)/priv
EX_FAISS_DIR = c_src/ex_faiss
EX_FAISS_CACHE_SO = cache/libex_faiss.so
EX_FAISS_CACHE_LIB_DIR = cache/lib
EX_FAISS_SO = $(PRIV_DIR)/libex_faiss.so
EX_FAISS_LIB_DIR = $(PRIV_DIR)/lib

# Build flags
CFLAGS = -I$(ERTS_INCLUDE_DIR) -I$(FAISS_DIR) -fPIC -O3 -shared -std=c++14
CMAKE_FLAGS = -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON

ifeq ($(USE_CUDA), true)
	CFLAGS += -D__CUDA__
	CMAKE_FLAGS += -DFAISS_ENABLE_GPU=ON
else
	CMAKE_FLAGS += -DFAISS_ENABLE_GPU=OFF
endif

C_SRCS = c_src/ex_faiss.cc $(EX_FAISS_DIR)/nif_util.cc $(EX_FAISS_DIR)/nif_util.h \
					$(EX_FAISS_DIR)/index.cc $(EX_FAISS_DIR)/index.h $(EX_FAISS_DIR)/clustering.cc \
					$(EX_FAISS_DIR)/clustering.h

LDFLAGS = -L$(EX_FAISS_CACHE_LIB_DIR) -lfaiss

ifeq ($(shell uname -s), Darwin)
	LDFLAGS += -flat_namespace -undefined suppress
	POST_INSTALL = install_name_tool $(EX_FAISS_CACHE_SO) -change @rpath/libfaiss.dylib @loader_path/lib/libfaiss.dylib

  ifeq ($(USE_LLVM_BREW), true)
		LLVM_PREFIX=$(shell brew --prefix llvm)

		CMAKE_FLAGS += -DCMAKE_CXX_COMPILER=$(LLVM_PREFIX)/bin/clang++
	endif
else
	# Use a relative RPATH, so at runtime libex_faiss.so looks for libfaiss.so
	# in ./lib regardless of the absolute location. This way priv can be safely
	# packed into an Elixir release. Also, we use $$ to escape Makefile variable
	# and single quotes to escape shell variable
	LDFLAGS += -Wl,-rpath,'$$ORIGIN/lib'
	POST_INSTALL = $(NOOP)
endif

$(EX_FAISS_SO): $(EX_FAISS_CACHE_SO)
	@ mkdir -p $(PRIV_DIR)
	@ if [ "${MIX_BUILD_EMBEDDED}" = "true" ]; then \
		cp -a $(abspath $(EX_FAISS_CACHE_LIB_DIR)) $(EX_FAISS_LIB_DIR) ; \
		cp -a $(abspath $(EX_FAISS_CACHE_SO)) $(EX_FAISS_SO) ; \
	else \
		ln -sf $(abspath $(EX_FAISS_CACHE_LIB_DIR)) $(EX_FAISS_LIB_DIR) ; \
		ln -sf $(abspath $(EX_FAISS_CACHE_SO)) $(EX_FAISS_SO) ; \
	fi

$(EX_FAISS_CACHE_SO): $(FAISS_LIB_DIR_FLAG) $(C_SRCS)
	@mkdir -p cache
	cp -a $(FAISS_LIB_DIR) $(EX_FAISS_CACHE_LIB_DIR)
	$(CXX) $(CFLAGS) c_src/ex_faiss.cc $(EX_FAISS_DIR)/nif_util.cc $(EX_FAISS_DIR)/index.cc \
		$(EX_FAISS_DIR)/clustering.cc -o $(EX_FAISS_CACHE_SO) $(LDFLAGS)
	$(POST_INSTALL)

$(FAISS_LIB_DIR_FLAG):
		rm -rf $(FAISS_DIR) && \
		mkdir -p $(FAISS_DIR) && \
			cd $(FAISS_DIR) && \
			git init && \
			git remote add origin $(FAISS_GIT_REPO) && \
			git fetch --depth 1 origin $(FAISS_GIT_REV) && \
			git checkout FETCH_HEAD && \
		  cmake -B build . $(CMAKE_FLAGS) && \
			make -C build -j faiss
		touch $(FAISS_LIB_DIR_FLAG)

clean:
	rm -rf $(EX_FAISS_CACHE_SO)
	rm -rf $(EX_FAISS_CACHE_LIB_DIR)
	rm -rf $(EX_FAISS_SO)
	rm -rf $(EX_FAISS_LIB_DIR)
	rm -rf $(FAISS_DIR)
