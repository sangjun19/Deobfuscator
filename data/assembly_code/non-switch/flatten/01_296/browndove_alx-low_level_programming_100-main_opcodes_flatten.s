	.file	"browndove_alx-low_level_programming_100-main_opcodes_flatten.c"
	.text
	.globl	_TIG_IZ_0RSf_envp
	.bss
	.align 8
	.type	_TIG_IZ_0RSf_envp, @object
	.size	_TIG_IZ_0RSf_envp, 8
_TIG_IZ_0RSf_envp:
	.zero	8
	.globl	_TIG_IZ_0RSf_argv
	.align 8
	.type	_TIG_IZ_0RSf_argv, @object
	.size	_TIG_IZ_0RSf_argv, 8
_TIG_IZ_0RSf_argv:
	.zero	8
	.globl	_TIG_IZ_0RSf_argc
	.align 4
	.type	_TIG_IZ_0RSf_argc, @object
	.size	_TIG_IZ_0RSf_argc, 4
_TIG_IZ_0RSf_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%.2hhx"
	.text
	.globl	print_opcodes
	.type	print_opcodes, @function
print_opcodes:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$1, -8(%rbp)
.L18:
	cmpq	$9, -8(%rbp)
	ja	.L19
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L19-.L4
	.long	.L11-.L4
	.long	.L20-.L4
	.long	.L9-.L4
	.long	.L19-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L5:
	movl	$10, %edi
	call	putchar@PLT
	movq	$2, -8(%rbp)
	jmp	.L12
.L11:
	movl	$0, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L12
.L9:
	movl	-28(%rbp), %eax
	subl	$1, %eax
	cmpl	%eax, -12(%rbp)
	jge	.L13
	movq	$7, -8(%rbp)
	jmp	.L12
.L13:
	movq	$9, -8(%rbp)
	jmp	.L12
.L3:
	addl	$1, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L12
.L7:
	movl	-12(%rbp), %eax
	cmpl	-28(%rbp), %eax
	jge	.L15
	movq	$5, -8(%rbp)
	jmp	.L12
.L15:
	movq	$8, -8(%rbp)
	jmp	.L12
.L8:
	movl	-12(%rbp), %eax
	movslq	%eax, %rdx
	movq	-24(%rbp), %rax
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L12
.L6:
	movl	$32, %edi
	call	putchar@PLT
	movq	$9, -8(%rbp)
	jmp	.L12
.L19:
	nop
.L12:
	jmp	.L18
.L20:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	print_opcodes, .-print_opcodes
	.section	.rodata
.LC1:
	.string	"Error"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, _TIG_IZ_0RSf_envp(%rip)
	nop
.L22:
	movq	$0, _TIG_IZ_0RSf_argv(%rip)
	nop
.L23:
	movl	$0, _TIG_IZ_0RSf_argc(%rip)
	nop
	nop
.L24:
.L25:
#APP
# 110 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-0RSf--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_0RSf_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_0RSf_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_0RSf_envp(%rip)
	nop
	movq	$4, -8(%rbp)
.L41:
	cmpq	$8, -8(%rbp)
	ja	.L43
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L28(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L28(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L28:
	.long	.L34-.L28
	.long	.L43-.L28
	.long	.L33-.L28
	.long	.L43-.L28
	.long	.L32-.L28
	.long	.L31-.L28
	.long	.L30-.L28
	.long	.L29-.L28
	.long	.L27-.L28
	.text
.L32:
	cmpl	$2, -20(%rbp)
	je	.L35
	movq	$6, -8(%rbp)
	jmp	.L37
.L35:
	movq	$8, -8(%rbp)
	jmp	.L37
.L27:
	movq	-32(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -12(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L37
.L30:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$1, %edi
	call	exit@PLT
.L31:
	movl	$0, %eax
	jmp	.L42
.L34:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$2, %edi
	call	exit@PLT
.L29:
	movl	-12(%rbp), %eax
	movl	%eax, %esi
	leaq	main(%rip), %rax
	movq	%rax, %rdi
	call	print_opcodes
	movq	$5, -8(%rbp)
	jmp	.L37
.L33:
	cmpl	$0, -12(%rbp)
	jns	.L39
	movq	$0, -8(%rbp)
	jmp	.L37
.L39:
	movq	$7, -8(%rbp)
	jmp	.L37
.L43:
	nop
.L37:
	jmp	.L41
.L42:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
