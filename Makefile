CC       ?= cc
CFLAGS   ?= -std=c11 -O2 -Wall -Wextra -Wpedantic
CPPFLAGS += -Iinclude

BUILD := build
OBJS  := $(patsubst src/%.c,$(BUILD)/%.o,$(wildcard src/*.c))
LIB   := $(BUILD)/libllm.a
TESTS := $(patsubst tests/%.c,$(BUILD)/%,$(wildcard tests/*.c))

GPT2_URL    := https://openaipublic.blob.core.windows.net/gpt-2/models/124M
VERDICT_URL := https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt
DATA        := data/gpt2/encoder.json data/gpt2/vocab.bpe data/the-verdict.txt

.PHONY: all test data clean

all: $(LIB)

$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

$(BUILD)/test_%: tests/test_%.c $(LIB) $(wildcard include/*.h)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LIB) -o $@

test: $(TESTS) $(DATA)
	@for t in $(TESTS); do ./$$t || exit 1; done

data: $(DATA)

data/gpt2/%:
	@mkdir -p $(@D)
	curl -fsSL -o $@ $(GPT2_URL)/$*

data/the-verdict.txt:
	@mkdir -p $(@D)
	curl -fsSL -o $@ $(VERDICT_URL)

$(BUILD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)

-include $(OBJS:.o=.d)
