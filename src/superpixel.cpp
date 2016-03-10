#include "superpixel.h"

void Superpixel::add_block(Block* block) {
    blocks[block->index] = block;
    sums += block->sums;
}

void Superpixel::remove_block(Block* block) {
    blocks.erase(block->index);
    sums -= block->sums;
}

Superpixel::Superpixel(Engine& engine, int id, int initial_size) : engine(engine), id(id), initial_size(initial_size), blocks() {
    sums = {};
}