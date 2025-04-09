	.file	"shivam-65_ss-assignments-2_1_flatten.c"
	.text
	.globl	_TIG_IZ_5iM6_envp
	.bss
	.align 8
	.type	_TIG_IZ_5iM6_envp, @object
	.size	_TIG_IZ_5iM6_envp, 8
_TIG_IZ_5iM6_envp:
	.zero	8
	.globl	_TIG_IZ_5iM6_argc
	.align 4
	.type	_TIG_IZ_5iM6_argc, @object
	.size	_TIG_IZ_5iM6_argc, 4
_TIG_IZ_5iM6_argc:
	.zero	4
	.globl	_TIG_IZ_5iM6_argv
	.align 8
	.type	_TIG_IZ_5iM6_argv, @object
	.size	_TIG_IZ_5iM6_argv, 8
_TIG_IZ_5iM6_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"Select interval timer\n1:10s\n2:10micros"
.LC1:
	.string	"%d"
.LC2:
	.string	"INVALID CHOICE"
	.align 8
.LC3:
	.string	"Error while setting an interval timer!"
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
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_5iM6_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_5iM6_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_5iM6_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 125 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-5iM6--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_5iM6_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_5iM6_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_5iM6_envp(%rip)
	nop
	movq	$11, -56(%rbp)
.L24:
	cmpq	$18, -56(%rbp)
	ja	.L26
	movq	-56(%rbp), %rax
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
	.long	.L17-.L8
	.long	.L26-.L8
	.long	.L26-.L8
	.long	.L26-.L8
	.long	.L26-.L8
	.long	.L26-.L8
	.long	.L16-.L8
	.long	.L26-.L8
	.long	.L26-.L8
	.long	.L15-.L8
	.long	.L26-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L26-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L7:
	leaq	-48(%rbp), %rax
	movl	$0, %edx
	movq	%rax, %rsi
	movl	$0, %edi
	call	setitimer@PLT
	movl	%eax, -60(%rbp)
	movl	-60(%rbp), %eax
	movl	%eax, -64(%rbp)
	movq	$13, -56(%rbp)
	jmp	.L18
.L11:
	movq	$0, -48(%rbp)
	movq	$0, -40(%rbp)
	movq	$0, -32(%rbp)
	movq	$10, -24(%rbp)
	movq	$18, -56(%rbp)
	jmp	.L18
.L10:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-68(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$17, -56(%rbp)
	jmp	.L18
.L13:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	_exit@PLT
.L14:
	movq	$15, -56(%rbp)
	jmp	.L18
.L15:
	movq	$0, -48(%rbp)
	movq	$0, -40(%rbp)
	movq	$10, -32(%rbp)
	movq	$0, -24(%rbp)
	movq	$18, -56(%rbp)
	jmp	.L18
.L12:
	cmpl	$-1, -64(%rbp)
	jne	.L19
	movq	$6, -56(%rbp)
	jmp	.L18
.L19:
	movq	$0, -56(%rbp)
	jmp	.L18
.L9:
	movl	-68(%rbp), %eax
	cmpl	$1, %eax
	je	.L21
	cmpl	$2, %eax
	jne	.L22
	movq	$14, -56(%rbp)
	jmp	.L23
.L21:
	movq	$9, -56(%rbp)
	jmp	.L23
.L22:
	movq	$12, -56(%rbp)
	nop
.L23:
	jmp	.L18
.L16:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$0, -56(%rbp)
	jmp	.L18
.L17:
	movq	$0, -56(%rbp)
	jmp	.L18
.L26:
	nop
.L18:
	jmp	.L24
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
