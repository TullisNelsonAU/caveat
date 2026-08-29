//! Interfaces with goblin to extract the `.text` section of an ELF. This module will help interact with the headers of ELF files and eventually other executable formats.
use anyhow::{Result, anyhow};
use goblin::Object;

/// Locate the `.text` section of an executable and returns its load address and bytes.
pub fn extract_text_section<'a>(buffer: &'a [u8]) -> Result<(u64, &'a [u8])> {
    match goblin::Object::parse(buffer)? {
        Object::Elf(elf) => {
            // Normal binaries: use the .text section when it's there.
            if let Some(text_hdr) = elf
                .section_headers
                .iter()
                .find(|s| elf.shdr_strtab.get_at(s.sh_name) == Some(".text"))
            {
                let range = text_hdr
                    .file_range()
                    .ok_or_else(|| anyhow!(".text has no file range"))?;
                return Ok((text_hdr.sh_addr, &buffer[range]));
            }
            // Headerless / packed: section headers got stripped (UPX, raw dumps, a lot of
            // malware). Sections are optional to a loader, so fall back to the first
            // executable PT_LOAD segment — that's the code region the OS actually maps.
            use goblin::elf::program_header::{PF_X, PT_LOAD};
            let seg = elf
                .program_headers
                .iter()
                .find(|p| p.p_type == PT_LOAD && p.p_flags & PF_X != 0)
                .ok_or_else(|| anyhow!("no .text section and no executable PT_LOAD segment"))?;
            let start = seg.p_offset as usize;
            let end = start
                .checked_add(seg.p_filesz as usize)
                .ok_or_else(|| anyhow!("executable segment size overflow"))?;
            if end > buffer.len() {
                return Err(anyhow!("executable segment runs past end of file"));
            }
            Ok((seg.p_vaddr, &buffer[start..end]))
        }
        Object::PE(pe) => {
            let text_hdr = pe
                .sections
                .iter()
                .find(|s| s.name().map_or(false, |n| n == ".text"))
                .ok_or_else(|| anyhow!(".text section not found"))?;
            let start = text_hdr.pointer_to_raw_data as usize;
            let end = start
                .checked_add(text_hdr.size_of_raw_data as usize)
                .ok_or_else(|| anyhow!(".text size overflow"))?;
            let load_address = pe.image_base as u64 + text_hdr.virtual_address as u64;
            Ok((load_address, &buffer[start..end]))
        }
        _ => Err(anyhow!("Unsupported binary format")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_text_section_elf() {
        let buffer = include_bytes!("../tests/bins/elf_test");
        let result = extract_text_section(buffer);
        assert!(result.is_ok());
    }

    #[test]
    fn test_extract_text_section_pe() {
        let buffer = include_bytes!("../tests/bins/pe_test.exe");
        let result = extract_text_section(buffer);
        assert!(result.is_ok());
    }

    #[test]
    fn test_extract_text_section_not_found() {
        let buffer = &[0u8; 64];
        let result = extract_text_section(buffer);
        assert!(result.is_err());
    }
}
