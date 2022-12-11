# Environment variables passed via elixir_make
# ERTS_INCLUDE_DIR
# MIX_APP_PATH

TEMP ?= $(HOME)/.cache
BUILD_CACHE ?= $(TEMP)/ex_faiss

FAISS_GIT_REPO ?= https://www.github.com/facebookresearch/faiss
FAISS_GIT_REV ?= 19f7696deedc93615c3ee0ff4de22284b53e0243
FAISS_NS = faiss-$(FAISS_GIT_REV)
FAISS_DIR = $(BUILD_CACHE)/$(FAISS_NS)

# Private configuration
EX_FAISS_DIR = c_src/ex_faiss
PRIV_DIR = $(MIX_APP_PATH)/priv
EX_FAISS_SO = $(PRIV_DIR)/libex_faiss.so

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

LDFLAGS = -L$(PRIV_DIR)/faiss -lfaiss

ifeq ($(shell uname -s), Darwin)
	LDFLAGS += -flat_namespace -undefined suppress
endif

$(EX_FAISS_SO): faiss $(C_SRCS)
	ln -sf $(FAISS_DIR)/build/faiss $(PRIV_DIR)/faiss
	$(CXX) $(CFLAGS) c_src/ex_faiss.cc $(EX_FAISS_DIR)/nif_util.cc $(EX_FAISS_DIR)/index.cc $(EX_FAISS_DIR)/clustering.cc -o $(EX_FAISS_SO) $(LDFLAGS)
	$(POST_INSTALL)

ifeq ($(shell test ! -d $(FAISS_DIR) && echo 1 || echo 0), 1)
faiss:
		rm -rf $(FAISS_DIR) && \
		mkdir -p $(FAISS_DIR) && \
			cd $(FAISS_DIR) && \
			git init && \
			git remote add origin $(FAISS_GIT_REPO) && \
			git fetch --depth 1 origin $(FAISS_GIT_REV) && \
			git checkout FETCH_HEAD && \
			cmake -B build . $(CMAKE_FLAGS) && \
			make -C build -j faiss
else
faiss:
	@echo "Using cached faiss build..."
endif

clean:
	rm -rf $(FAISS_DIR)
	rm -rf $(EX_FAISS_SO)