	.file	"hitaarthh_C-Programming_gcd_flatten.c"
	.text
	.globl	_TIG_IZ_G5KK_argv
	.bss
	.align 8
	.type	_TIG_IZ_G5KK_argv, @object
	.size	_TIG_IZ_G5KK_argv, 8
_TIG_IZ_G5KK_argv:
	.zero	8
	.globl	_TIG_IZ_G5KK_envp
	.align 8
	.type	_TIG_IZ_G5KK_envp, @object
	.size	_TIG_IZ_G5KK_envp, 8
_TIG_IZ_G5KK_envp:
	.zero	8
	.globl	_TIG_IZ_G5KK_argc
	.align 4
	.type	_TIG_IZ_G5KK_argc, @object
	.size	_TIG_IZ_G5KK_argc, 4
_TIG_IZ_G5KK_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"G.C.D of %d and %d is %d"
.LC1:
	.string	"Enter two integers: "
.LC2:
	.string	"%d %d"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_G5KK_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_G5KK_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_G5KK_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 103 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-G5KK--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_G5KK_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_G5KK_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_G5KK_envp(%rip)
	nop
	movq	$7, -16(%rbp)
.L28:
	cmpq	$12, -16(%rbp)
	ja	.L31
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
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L31-.L8
	.long	.L15-.L8
	.long	.L31-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L31-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L7:
	movl	-32(%rbp), %eax
	cmpl	%eax, -24(%rbp)
	jg	.L18
	movq	$5, -16(%rbp)
	jmp	.L20
.L18:
	movq	$8, -16(%rbp)
	jmp	.L20
.L11:
	movl	-28(%rbp), %edx
	movl	-32(%rbp), %eax
	movl	-20(%rbp), %ecx
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$1, -16(%rbp)
	jmp	.L20
.L16:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L29
	jmp	.L30
.L15:
	addl	$1, -24(%rbp)
	movq	$12, -16(%rbp)
	jmp	.L20
.L9:
	movl	-32(%rbp), %eax
	cltd
	idivl	-24(%rbp)
	movl	%edx, %eax
	testl	%eax, %eax
	jne	.L22
	movq	$9, -16(%rbp)
	jmp	.L20
.L22:
	movq	$3, -16(%rbp)
	jmp	.L20
.L10:
	movl	-28(%rbp), %eax
	cltd
	idivl	-24(%rbp)
	movl	%edx, %eax
	testl	%eax, %eax
	jne	.L24
	movq	$0, -16(%rbp)
	jmp	.L20
.L24:
	movq	$3, -16(%rbp)
	jmp	.L20
.L13:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rdx
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$1, -24(%rbp)
	movq	$12, -16(%rbp)
	jmp	.L20
.L14:
	movl	-28(%rbp), %eax
	cmpl	%eax, -24(%rbp)
	jg	.L26
	movq	$11, -16(%rbp)
	jmp	.L20
.L26:
	movq	$8, -16(%rbp)
	jmp	.L20
.L17:
	movl	-24(%rbp), %eax
	movl	%eax, -20(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L20
.L12:
	movq	$6, -16(%rbp)
	jmp	.L20
.L31:
	nop
.L20:
	jmp	.L28
.L30:
	call	__stack_chk_fail@PLT
.L29:
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
