use std::io::BufRead;

use crate::decoder::DecodingError;

use super::lossless::BitReader;

/// Rudimentary utility for reading Canonical Huffman Codes.
/// Based off https://github.com/webmproject/libwebp/blob/7f8472a610b61ec780ef0a8873cd954ac512a505/src/utils/huffman.c
///

const MAX_ALLOWED_CODE_LENGTH: usize = 15;
const TABLE_BITS: u8 = 10;
const TABLE_MASK: u16 = (1 << TABLE_BITS) - 1;
const SECONDARY_TABLE_ENTRY: u32 = 1 << 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HuffmanTreeNode {
    Branch(usize), //offset in vector to children
    Leaf(u16),     //symbol stored in leaf
    Empty,
}

#[derive(Clone, Debug)]
enum HuffmanTreeInner {
    Single(u16),
    Tree {
        primary_table: [u32; 1 << TABLE_BITS],
        secondary_table: Vec<u16>,
    },
}

/// Return the next code, or if the codeword is already all ones (which is the final code), return
/// the same code again.
fn next_codeword(mut codeword: u16, table_size: u16) -> u16 {
    if codeword == table_size - 1 {
        return codeword;
    }

    let adv = (u16::BITS - 1) - (codeword ^ (table_size - 1)).leading_zeros();
    let bit = 1 << adv;
    codeword &= bit - 1;
    codeword |= bit;
    codeword
}

/// Huffman tree
#[derive(Clone, Debug)]
pub(crate) struct HuffmanTree(HuffmanTreeInner);

impl Default for HuffmanTree {
    fn default() -> Self {
        Self(HuffmanTreeInner::Single(0))
    }
}

impl HuffmanTree {
    /// Builds a tree implicitly, just from code lengths
    pub(crate) fn build_implicit(code_lengths: &[u8]) -> Result<HuffmanTree, DecodingError> {
        // Count symbols and build histogram
        let mut histogram = [0; 16];
        for &length in code_lengths {
            histogram[length as usize] += 1;
        }

        // Handle special cases
        if histogram[0] == code_lengths.len() {
            return Err(DecodingError::HuffmanError);
        } else if histogram[0] == code_lengths.len() - 1 {
            let root_symbol = code_lengths.iter().position(|&x| x != 0).unwrap() as u16;
            return Ok(Self::build_single_node(root_symbol));
        };

        // Determine the maximum code length.
        let mut max_length = 15;
        while max_length > 1 && histogram[max_length] == 0 {
            max_length -= 1;
        }

        // Sort symbols by code length. Given the histogram, we can determine the starting offset
        // for each code length.
        let mut offsets = [0; 16];
        let mut codespace_used = 0usize;
        offsets[1] = histogram[0];
        for i in 1..max_length {
            offsets[i + 1] = offsets[i] + histogram[i];
            codespace_used = (codespace_used << 1) + histogram[i];
        }
        codespace_used = (codespace_used << 1) + histogram[max_length];

        // Confirm that the huffman tree is valid
        if codespace_used != 1 << max_length {
            return Err(DecodingError::HuffmanError);
        }

        // Sort the symbols by code length.
        let mut next_index = offsets;
        let mut sorted_symbols = vec![0; code_lengths.len()];
        for symbol in 0..code_lengths.len() {
            let length = code_lengths[symbol];
            sorted_symbols[next_index[length as usize]] = symbol;
            next_index[length as usize] += 1;
        }

        let mut codeword = 0u16;
        let mut i = histogram[0];

        // Populate the primary decoding table
        let mut primary_table = [0u32; 1 << TABLE_BITS];
        let primary_table_bits = primary_table.len().ilog2() as usize;
        let primary_table_mask = (1 << primary_table_bits) - 1;
        for length in 1..=primary_table_bits {
            let current_table_end = 1 << length;

            // Loop over all symbols with the current code length and set their table entries.
            for _ in 0..histogram[length] {
                let symbol = sorted_symbols[i];
                i += 1;

                primary_table[codeword as usize] = ((symbol as u32) << 16) | length as u32;
                codeword = next_codeword(codeword, current_table_end as u16);
            }

            // If we aren't at the maximum table size, double the size of the table.
            if length < primary_table_bits {
                primary_table.copy_within(0..current_table_end, current_table_end);
            }
        }

        // Populate the secondary decoding table.
        let mut secondary_table = Vec::new();
        if max_length > primary_table_bits {
            let mut subtable_start = 0;
            let mut subtable_prefix = !0;
            for length in (primary_table_bits + 1)..=max_length {
                let subtable_size = 1 << (length - primary_table_bits);
                for _ in 0..histogram[length] {
                    // If the codeword's prefix doesn't match the current subtable, create a new
                    // subtable.
                    if codeword & primary_table_mask != subtable_prefix {
                        subtable_prefix = codeword & primary_table_mask;
                        subtable_start = secondary_table.len();
                        primary_table[subtable_prefix as usize] = ((subtable_start as u32) << 16)
                            | SECONDARY_TABLE_ENTRY
                            | (subtable_size as u32 - 1);
                        secondary_table.resize(subtable_start + subtable_size, 0);
                    }

                    // Lookup the symbol.
                    let symbol = sorted_symbols[i];
                    i += 1;

                    // Insert the symbol into the secondary table and advance to the next codeword.
                    secondary_table[subtable_start + (codeword >> primary_table_bits) as usize] =
                        ((symbol as u16) << 4) | (length as u16);
                    codeword = next_codeword(codeword, 1 << length);
                }

                // If there are more codes with the same subtable prefix, extend the subtable.
                if length < max_length && codeword & primary_table_mask == subtable_prefix {
                    secondary_table.extend_from_within(subtable_start..);
                    let subtable_size = secondary_table.len() - subtable_start;
                    primary_table[subtable_prefix as usize] = ((subtable_start as u32) << 16)
                        | SECONDARY_TABLE_ENTRY
                        | (subtable_size as u32 - 1);
                }
            }
        }

        Ok(Self(HuffmanTreeInner::Tree {
            primary_table,
            secondary_table,
        }))
    }

    pub(crate) fn build_single_node(symbol: u16) -> HuffmanTree {
        Self(HuffmanTreeInner::Single(symbol))
    }

    pub(crate) fn build_two_node(zero: u16, one: u16) -> HuffmanTree {
        let mut primary_table = [0u32; 1 << TABLE_BITS];
        for pair in primary_table.chunks_exact_mut(2) {
            pair[0] = ((zero as u32) << 16) | 1;
            pair[1] = ((one as u32) << 16) | 1;
        }

        Self(HuffmanTreeInner::Tree {
            primary_table,
            secondary_table: Vec::new(),
        })
    }

    pub(crate) fn is_single_node(&self) -> bool {
        matches!(self.0, HuffmanTreeInner::Single(_))
    }

    /// Reads a symbol using the bit reader.
    ///
    /// You must call call `bit_reader.fill()` before calling this function or it may erroroneosly
    /// detect the end of the stream and return a bitstream error.
    pub(crate) fn read_symbol<R: BufRead>(
        &self,
        bit_reader: &mut BitReader<R>,
    ) -> Result<u16, DecodingError> {
        match &self.0 {
            HuffmanTreeInner::Tree {
                primary_table,
                secondary_table,
            } => {
                let v = bit_reader.peek_full() as u16;
                let entry = primary_table[(v & TABLE_MASK) as usize];
                if entry & SECONDARY_TABLE_ENTRY == 0 {
                    bit_reader.consume(entry as u8)?;
                    return Ok((entry >> 16) as u16);
                } else {
                    let subtable_start = (entry >> 16) as usize;
                    let subtable_mask = (entry & 0xff) as u16;
                    let subtable_entry = secondary_table
                        [subtable_start + ((v >> TABLE_BITS) & subtable_mask) as usize];
                    bit_reader.consume((subtable_entry & 0xf) as u8)?;
                    return Ok(subtable_entry >> 4);
                }
            }
            HuffmanTreeInner::Single(symbol) => Ok(*symbol),
        }
    }

    /// Peek at the next symbol in the bitstream if it can be read with only a primary table lookup.
    ///
    /// Returns a tuple of the codelength and symbol value. This function may return wrong
    /// information if there aren't enough bits in the bit reader to read the next symbol.
    pub(crate) fn peek_symbol<R: BufRead>(
        &self,
        bit_reader: &mut BitReader<R>,
    ) -> Option<(u8, u16)> {
        match &self.0 {
            HuffmanTreeInner::Tree {
                primary_table, ..
            } => {
                let v = bit_reader.peek_full() as u16;
                let entry = primary_table[(v & TABLE_MASK) as usize];
                if entry & SECONDARY_TABLE_ENTRY == 0 {
                    return Some((entry as u8, (entry >> 16) as u16));
                }
                None
            }
            HuffmanTreeInner::Single(symbol) => Some((0, *symbol)),
        }
    }
}
