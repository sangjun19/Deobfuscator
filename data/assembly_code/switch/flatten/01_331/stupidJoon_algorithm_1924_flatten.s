	.file	"stupidJoon_algorithm_1924_flatten.c"
	.text
	.globl	_TIG_IZ_ieUj_envp
	.bss
	.align 8
	.type	_TIG_IZ_ieUj_envp, @object
	.size	_TIG_IZ_ieUj_envp, 8
_TIG_IZ_ieUj_envp:
	.zero	8
	.globl	_TIG_IZ_ieUj_argv
	.align 8
	.type	_TIG_IZ_ieUj_argv, @object
	.size	_TIG_IZ_ieUj_argv, 8
_TIG_IZ_ieUj_argv:
	.zero	8
	.globl	_TIG_IZ_ieUj_argc
	.align 4
	.type	_TIG_IZ_ieUj_argc, @object
	.size	_TIG_IZ_ieUj_argc, 4
_TIG_IZ_ieUj_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%d %d"
.LC1:
	.string	"FRI"
.LC2:
	.string	"MON"
.LC3:
	.string	"SUN"
.LC4:
	.string	"TUE"
.LC5:
	.string	"WED"
.LC6:
	.string	"SAT"
.LC7:
	.string	"THU"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
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
	movq	$0, _TIG_IZ_ieUj_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_ieUj_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_ieUj_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 103 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-ieUj--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_ieUj_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_ieUj_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_ieUj_envp(%rip)
	nop
	movq	$2, -16(%rbp)
.L57:
	cmpq	$33, -16(%rbp)
	ja	.L60
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
	.long	.L32-.L8
	.long	.L31-.L8
	.long	.L30-.L8
	.long	.L60-.L8
	.long	.L60-.L8
	.long	.L29-.L8
	.long	.L60-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L26-.L8
	.long	.L60-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L60-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L60-.L8
	.long	.L15-.L8
	.long	.L60-.L8
	.long	.L60-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L60-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L19:
	leaq	-28(%rbp), %rdx
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-28(%rbp), %eax
	movl	%eax, -24(%rbp)
	movl	$1, -20(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L33
.L11:
	addl	$28, -24(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L33
.L22:
	addl	$30, -24(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L33
.L21:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$31, -16(%rbp)
	jmp	.L33
.L10:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L58
	jmp	.L59
.L24:
	addl	$30, -24(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L33
.L27:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$31, -16(%rbp)
	jmp	.L33
.L31:
	addl	$31, -24(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L33
.L15:
	addl	$30, -24(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L33
.L16:
	addl	$30, -24(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L33
.L14:
	cmpl	$9, -20(%rbp)
	jne	.L35
	movq	$14, -16(%rbp)
	jmp	.L33
.L35:
	movq	$32, -16(%rbp)
	jmp	.L33
.L25:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$31, -16(%rbp)
	jmp	.L33
.L26:
	addl	$1, -20(%rbp)
	movq	$0, -16(%rbp)
	jmp	.L33
.L23:
	cmpl	$2, -20(%rbp)
	jne	.L37
	movq	$30, -16(%rbp)
	jmp	.L33
.L37:
	movq	$19, -16(%rbp)
	jmp	.L33
.L18:
	cmpl	$4, -20(%rbp)
	jne	.L39
	movq	$23, -16(%rbp)
	jmp	.L33
.L39:
	movq	$27, -16(%rbp)
	jmp	.L33
.L9:
	cmpl	$11, -20(%rbp)
	jne	.L41
	movq	$21, -16(%rbp)
	jmp	.L33
.L41:
	movq	$1, -16(%rbp)
	jmp	.L33
.L20:
	movl	-24(%rbp), %edx
	movslq	%edx, %rax
	imulq	$-1840700269, %rax, %rax
	shrq	$32, %rax
	addl	%edx, %eax
	sarl	$2, %eax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	movl	%eax, %ecx
	sall	$3, %ecx
	subl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	cmpl	$6, %eax
	ja	.L43
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L45(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L45(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L45:
	.long	.L51-.L45
	.long	.L50-.L45
	.long	.L49-.L45
	.long	.L48-.L45
	.long	.L47-.L45
	.long	.L46-.L45
	.long	.L44-.L45
	.text
.L44:
	movq	$29, -16(%rbp)
	jmp	.L52
.L46:
	movq	$15, -16(%rbp)
	jmp	.L52
.L47:
	movq	$20, -16(%rbp)
	jmp	.L52
.L48:
	movq	$7, -16(%rbp)
	jmp	.L52
.L49:
	movq	$33, -16(%rbp)
	jmp	.L52
.L50:
	movq	$8, -16(%rbp)
	jmp	.L52
.L51:
	movq	$11, -16(%rbp)
	jmp	.L52
.L43:
	movq	$5, -16(%rbp)
	nop
.L52:
	jmp	.L33
.L13:
	cmpl	$6, -20(%rbp)
	jne	.L53
	movq	$12, -16(%rbp)
	jmp	.L33
.L53:
	movq	$26, -16(%rbp)
	jmp	.L33
.L29:
	movq	$31, -16(%rbp)
	jmp	.L33
.L7:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$31, -16(%rbp)
	jmp	.L33
.L32:
	movl	-32(%rbp), %eax
	cmpl	%eax, -20(%rbp)
	jge	.L55
	movq	$13, -16(%rbp)
	jmp	.L33
.L55:
	movq	$17, -16(%rbp)
	jmp	.L33
.L28:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$31, -16(%rbp)
	jmp	.L33
.L12:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$31, -16(%rbp)
	jmp	.L33
.L30:
	movq	$18, -16(%rbp)
	jmp	.L33
.L17:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$31, -16(%rbp)
	jmp	.L33
.L60:
	nop
.L33:
	jmp	.L57
.L59:
	call	__stack_chk_fail@PLT
.L58:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
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
