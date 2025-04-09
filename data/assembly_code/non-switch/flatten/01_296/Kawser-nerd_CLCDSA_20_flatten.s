	.file	"Kawser-nerd_CLCDSA_20_flatten.c"
	.text
	.globl	_TIG_IZ_SfP4_envp
	.bss
	.align 8
	.type	_TIG_IZ_SfP4_envp, @object
	.size	_TIG_IZ_SfP4_envp, 8
_TIG_IZ_SfP4_envp:
	.zero	8
	.globl	_TIG_IZ_SfP4_argc
	.align 4
	.type	_TIG_IZ_SfP4_argc, @object
	.size	_TIG_IZ_SfP4_argc, 4
_TIG_IZ_SfP4_argc:
	.zero	4
	.globl	_TIG_IZ_SfP4_argv
	.align 8
	.type	_TIG_IZ_SfP4_argv, @object
	.size	_TIG_IZ_SfP4_argv, 8
_TIG_IZ_SfP4_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%u"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_SfP4_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_SfP4_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_SfP4_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 104 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-SfP4--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_SfP4_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_SfP4_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_SfP4_envp(%rip)
	nop
	movq	$4, -16(%rbp)
.L17:
	cmpq	$7, -16(%rbp)
	ja	.L20
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
	.long	.L20-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L20-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L20-.L8
	.long	.L7-.L8
	.text
.L10:
	movq	$5, -16(%rbp)
	jmp	.L13
.L12:
	movl	-24(%rbp), %eax
	cmpl	%eax, -20(%rbp)
	ja	.L14
	movq	$7, -16(%rbp)
	jmp	.L13
.L14:
	movq	$2, -16(%rbp)
	jmp	.L13
.L9:
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$1, -20(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L13
.L7:
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	doit
	addl	$1, -20(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L13
.L11:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L18
	jmp	.L19
.L20:
	nop
.L13:
	jmp	.L17
.L19:
	call	__stack_chk_fail@PLT
.L18:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.section	.rodata
.LC1:
	.string	"%u %u"
.LC2:
	.string	"Case #%u: ON\n"
.LC3:
	.string	"Case #%u: OFF\n"
	.text
	.globl	doit
	.type	doit, @function
doit:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$1, -16(%rbp)
.L34:
	cmpq	$5, -16(%rbp)
	ja	.L37
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L24(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L24(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L24:
	.long	.L29-.L24
	.long	.L28-.L24
	.long	.L27-.L24
	.long	.L38-.L24
	.long	.L25-.L24
	.long	.L23-.L24
	.text
.L25:
	leaq	-24(%rbp), %rdx
	leaq	-28(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-28(%rbp), %edx
	movl	$32, %eax
	subl	%edx, %eax
	movl	$-1, %edx
	movl	%eax, %ecx
	shrl	%cl, %edx
	movl	%edx, %eax
	movl	%eax, -20(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L30
.L28:
	movq	$4, -16(%rbp)
	jmp	.L30
.L23:
	movl	-36(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L30
.L29:
	movl	-36(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -16(%rbp)
	jmp	.L30
.L27:
	movl	-24(%rbp), %eax
	andl	-20(%rbp), %eax
	cmpl	%eax, -20(%rbp)
	jne	.L32
	movq	$5, -16(%rbp)
	jmp	.L30
.L32:
	movq	$0, -16(%rbp)
	jmp	.L30
.L37:
	nop
.L30:
	jmp	.L34
.L38:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L36
	call	__stack_chk_fail@PLT
.L36:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	doit, .-doit
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
