	.file	"bcalagoz_codeforwin_c_exercises_exercise21_flatten.c"
	.text
	.globl	_TIG_IZ_CM1N_argc
	.bss
	.align 4
	.type	_TIG_IZ_CM1N_argc, @object
	.size	_TIG_IZ_CM1N_argc, 4
_TIG_IZ_CM1N_argc:
	.zero	4
	.globl	_TIG_IZ_CM1N_envp
	.align 8
	.type	_TIG_IZ_CM1N_envp, @object
	.size	_TIG_IZ_CM1N_envp, 8
_TIG_IZ_CM1N_envp:
	.zero	8
	.globl	_TIG_IZ_CM1N_argv
	.align 8
	.type	_TIG_IZ_CM1N_argv, @object
	.size	_TIG_IZ_CM1N_argv, 8
_TIG_IZ_CM1N_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Enter rows:"
.LC1:
	.string	"%d"
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
	movq	$0, _TIG_IZ_CM1N_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_CM1N_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_CM1N_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 108 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-CM1N--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_CM1N_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_CM1N_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_CM1N_envp(%rip)
	nop
	movq	$7, -16(%rbp)
.L33:
	cmpq	$22, -16(%rbp)
	ja	.L36
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
	.long	.L22-.L8
	.long	.L36-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L36-.L8
	.long	.L17-.L8
	.long	.L36-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L36-.L8
	.long	.L36-.L8
	.long	.L13-.L8
	.long	.L36-.L8
	.long	.L12-.L8
	.long	.L36-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L36-.L8
	.long	.L7-.L8
	.text
.L11:
	addl	$1, -28(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L23
.L19:
	movl	-20(%rbp), %eax
	cmpl	-32(%rbp), %eax
	jg	.L24
	movq	$16, -16(%rbp)
	jmp	.L23
.L24:
	movq	$2, -16(%rbp)
	jmp	.L23
.L13:
	subl	$1, -32(%rbp)
	addl	$1, -36(%rbp)
	movq	$18, -16(%rbp)
	jmp	.L23
.L20:
	movl	$1, -24(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L23
.L12:
	movl	$42, %edi
	call	putchar@PLT
	addl	$1, -20(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L23
.L14:
	movl	-40(%rbp), %eax
	cmpl	%eax, -28(%rbp)
	jge	.L26
	movq	$22, -16(%rbp)
	jmp	.L23
.L26:
	movq	$14, -16(%rbp)
	jmp	.L23
.L16:
	movl	-40(%rbp), %eax
	addl	%eax, %eax
	cmpl	%eax, -28(%rbp)
	jge	.L28
	movq	$3, -16(%rbp)
	jmp	.L23
.L28:
	movq	$5, -16(%rbp)
	jmp	.L23
.L10:
	movl	$32, %edi
	call	putchar@PLT
	addl	$1, -24(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L23
.L7:
	addl	$1, -32(%rbp)
	subl	$1, -36(%rbp)
	movq	$18, -16(%rbp)
	jmp	.L23
.L18:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L34
	jmp	.L35
.L15:
	movl	-24(%rbp), %eax
	cmpl	-36(%rbp), %eax
	jg	.L31
	movq	$19, -16(%rbp)
	jmp	.L23
.L31:
	movq	$20, -16(%rbp)
	jmp	.L23
.L22:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-40(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-40(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -36(%rbp)
	movl	$1, -32(%rbp)
	movl	$1, -28(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L23
.L17:
	movq	$0, -16(%rbp)
	jmp	.L23
.L21:
	movl	$10, %edi
	call	putchar@PLT
	movq	$11, -16(%rbp)
	jmp	.L23
.L9:
	movl	$1, -20(%rbp)
	movq	$4, -16(%rbp)
	jmp	.L23
.L36:
	nop
.L23:
	jmp	.L33
.L35:
	call	__stack_chk_fail@PLT
.L34:
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
