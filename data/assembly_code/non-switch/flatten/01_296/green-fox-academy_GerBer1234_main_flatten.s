	.file	"green-fox-academy_GerBer1234_main_flatten.c"
	.text
	.globl	_TIG_IZ_VOri_argc
	.bss
	.align 4
	.type	_TIG_IZ_VOri_argc, @object
	.size	_TIG_IZ_VOri_argc, 4
_TIG_IZ_VOri_argc:
	.zero	4
	.globl	_TIG_IZ_VOri_argv
	.align 8
	.type	_TIG_IZ_VOri_argv, @object
	.size	_TIG_IZ_VOri_argv, 8
_TIG_IZ_VOri_argv:
	.zero	8
	.globl	_TIG_IZ_VOri_envp
	.align 8
	.type	_TIG_IZ_VOri_envp, @object
	.size	_TIG_IZ_VOri_envp, 8
_TIG_IZ_VOri_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Number %d: %d\n"
.LC1:
	.string	"Error! memory not allocated."
.LC2:
	.string	"%d"
.LC3:
	.string	"Your number is: "
.LC4:
	.string	"Give me %d numbers, please\n"
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_VOri_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_VOri_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_VOri_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 108 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-VOri--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_VOri_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_VOri_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_VOri_envp(%rip)
	nop
	movq	$18, -24(%rbp)
.L29:
	cmpq	$20, -24(%rbp)
	ja	.L32
	movq	-24(%rbp), %rax
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
	.long	.L32-.L8
	.long	.L32-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L32-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L32-.L8
	.long	.L32-.L8
	.long	.L32-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L32-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L32-.L8
	.long	.L7-.L8
	.text
.L9:
	movq	$3, -24(%rbp)
	jmp	.L21
.L18:
	movl	$-1, %eax
	jmp	.L30
.L12:
	movl	-36(%rbp), %eax
	leal	1(%rax), %edx
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-32(%rbp), %rax
	addq	%rcx, %rax
	addl	%edx, %edx
	movl	%edx, (%rax)
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	-36(%rbp), %edx
	leal	1(%rdx), %ecx
	movl	%eax, %edx
	movl	%ecx, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -36(%rbp)
	movq	$10, -24(%rbp)
	jmp	.L21
.L11:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$4, -24(%rbp)
	jmp	.L21
.L15:
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	addl	%eax, -44(%rbp)
	addl	$1, -40(%rbp)
	movq	$6, -24(%rbp)
	jmp	.L21
.L19:
	movl	$0, -44(%rbp)
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$0, -32(%rbp)
	movl	-48(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$9, -24(%rbp)
	jmp	.L21
.L14:
	cmpq	$0, -32(%rbp)
	jne	.L23
	movq	$15, -24(%rbp)
	jmp	.L21
.L23:
	movq	$20, -24(%rbp)
	jmp	.L21
.L10:
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$5, -24(%rbp)
	jmp	.L21
.L16:
	movl	-48(%rbp), %eax
	cmpl	%eax, -40(%rbp)
	jge	.L25
	movq	$8, -24(%rbp)
	jmp	.L21
.L25:
	movq	$2, -24(%rbp)
	jmp	.L21
.L17:
	movl	$0, %eax
	jmp	.L30
.L13:
	movl	-36(%rbp), %eax
	cmpl	-44(%rbp), %eax
	jge	.L27
	movq	$14, -24(%rbp)
	jmp	.L21
.L27:
	movq	$17, -24(%rbp)
	jmp	.L21
.L20:
	movl	-44(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movl	$0, -36(%rbp)
	movq	$10, -24(%rbp)
	jmp	.L21
.L7:
	movl	-48(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -40(%rbp)
	movq	$6, -24(%rbp)
	jmp	.L21
.L32:
	nop
.L21:
	jmp	.L29
.L30:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L31
	call	__stack_chk_fail@PLT
.L31:
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
