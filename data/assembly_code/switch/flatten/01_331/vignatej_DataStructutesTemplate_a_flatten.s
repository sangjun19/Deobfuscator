	.file	"vignatej_DataStructutesTemplate_a_flatten.c"
	.text
	.globl	_TIG_IZ_qeF5_envp
	.bss
	.align 8
	.type	_TIG_IZ_qeF5_envp, @object
	.size	_TIG_IZ_qeF5_envp, 8
_TIG_IZ_qeF5_envp:
	.zero	8
	.globl	_TIG_IZ_qeF5_argv
	.align 8
	.type	_TIG_IZ_qeF5_argv, @object
	.size	_TIG_IZ_qeF5_argv, 8
_TIG_IZ_qeF5_argv:
	.zero	8
	.globl	_TIG_IZ_qeF5_argc
	.align 4
	.type	_TIG_IZ_qeF5_argc, @object
	.size	_TIG_IZ_qeF5_argc, 4
_TIG_IZ_qeF5_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%d, hello\n"
.LC1:
	.string	"fir"
.LC2:
	.string	"def"
.LC3:
	.string	"sec"
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	$0, _TIG_IZ_qeF5_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_qeF5_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_qeF5_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 108 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-qeF5--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_qeF5_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_qeF5_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_qeF5_envp(%rip)
	nop
	movq	$1, -16(%rbp)
.L20:
	cmpq	$9, -16(%rbp)
	ja	.L22
	movq	-16(%rbp), %rax
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
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L22-.L8
	.long	.L9-.L8
	.long	.L22-.L8
	.long	.L22-.L8
	.long	.L7-.L8
	.text
.L10:
	movl	$2, -32(%rbp)
	movl	$5, -28(%rbp)
	movl	$0, %edi
	call	time@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	%eax, %edi
	call	srand@PLT
	call	rand@PLT
	movl	%eax, -32(%rbp)
	addl	$1, -32(%rbp)
	movl	-32(%rbp), %eax
	addl	$5, %eax
	movl	%eax, -24(%rbp)
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	call	getchar@PLT
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movb	%al, -33(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L15
.L13:
	movq	$4, -16(%rbp)
	jmp	.L15
.L11:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$9, -16(%rbp)
	jmp	.L15
.L7:
	movl	$0, %eax
	jmp	.L21
.L9:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$9, -16(%rbp)
	jmp	.L15
.L14:
	movsbl	-33(%rbp), %eax
	cmpl	$97, %eax
	je	.L17
	cmpl	$98, %eax
	jne	.L18
	movq	$2, -16(%rbp)
	jmp	.L19
.L17:
	movq	$3, -16(%rbp)
	jmp	.L19
.L18:
	movq	$6, -16(%rbp)
	nop
.L19:
	jmp	.L15
.L12:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$9, -16(%rbp)
	jmp	.L15
.L22:
	nop
.L15:
	jmp	.L20
.L21:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
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
