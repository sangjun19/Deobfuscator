	.file	"001ryu-ryu_C-learning_switch_flatten.c"
	.text
	.globl	_TIG_IZ_cXFE_argv
	.bss
	.align 8
	.type	_TIG_IZ_cXFE_argv, @object
	.size	_TIG_IZ_cXFE_argv, 8
_TIG_IZ_cXFE_argv:
	.zero	8
	.globl	_TIG_IZ_cXFE_envp
	.align 8
	.type	_TIG_IZ_cXFE_envp, @object
	.size	_TIG_IZ_cXFE_envp, 8
_TIG_IZ_cXFE_envp:
	.zero	8
	.globl	_TIG_IZ_cXFE_argc
	.align 4
	.type	_TIG_IZ_cXFE_argc, @object
	.size	_TIG_IZ_cXFE_argc, 4
_TIG_IZ_cXFE_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Wednesday"
.LC1:
	.string	"Saturday"
.LC2:
	.string	"Thursday"
.LC3:
	.string	"Sunday"
.LC4:
	.string	"Monday"
.LC5:
	.string	"Tuesday"
.LC6:
	.string	"Does not exist"
.LC7:
	.string	"Friday"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
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
	movq	$0, _TIG_IZ_cXFE_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_cXFE_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_cXFE_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 107 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-cXFE--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_cXFE_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_cXFE_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_cXFE_envp(%rip)
	nop
	movq	$4, -8(%rbp)
.L31:
	cmpq	$17, -8(%rbp)
	ja	.L33
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L8(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L8(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L8:
	.long	.L33-.L8
	.long	.L18-.L8
	.long	.L33-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L33-.L8
	.long	.L33-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L33-.L8
	.long	.L10-.L8
	.long	.L33-.L8
	.long	.L33-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L16:
	movl	$8, -12(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L19
.L18:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L19
.L17:
	movl	$0, %eax
	jmp	.L32
.L9:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L19
.L11:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L19
.L10:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L19
.L7:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L19
.L14:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L19
.L15:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L19
.L12:
	cmpl	$7, -12(%rbp)
	ja	.L21
	movl	-12(%rbp), %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L23(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L23(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L23:
	.long	.L21-.L23
	.long	.L29-.L23
	.long	.L28-.L23
	.long	.L27-.L23
	.long	.L26-.L23
	.long	.L25-.L23
	.long	.L24-.L23
	.long	.L22-.L23
	.text
.L22:
	movq	$13, -8(%rbp)
	jmp	.L30
.L24:
	movq	$16, -8(%rbp)
	jmp	.L30
.L25:
	movq	$7, -8(%rbp)
	jmp	.L30
.L26:
	movq	$11, -8(%rbp)
	jmp	.L30
.L27:
	movq	$1, -8(%rbp)
	jmp	.L30
.L28:
	movq	$6, -8(%rbp)
	jmp	.L30
.L29:
	movq	$17, -8(%rbp)
	jmp	.L30
.L21:
	movq	$5, -8(%rbp)
	nop
.L30:
	jmp	.L19
.L13:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -8(%rbp)
	jmp	.L19
.L33:
	nop
.L19:
	jmp	.L31
.L32:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
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
