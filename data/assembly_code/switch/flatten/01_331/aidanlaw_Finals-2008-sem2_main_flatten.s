	.file	"aidanlaw_Finals-2008-sem2_main_flatten.c"
	.text
	.globl	_TIG_IZ_VA7E_envp
	.bss
	.align 8
	.type	_TIG_IZ_VA7E_envp, @object
	.size	_TIG_IZ_VA7E_envp, 8
_TIG_IZ_VA7E_envp:
	.zero	8
	.globl	_TIG_IZ_VA7E_argv
	.align 8
	.type	_TIG_IZ_VA7E_argv, @object
	.size	_TIG_IZ_VA7E_argv, 8
_TIG_IZ_VA7E_argv:
	.zero	8
	.globl	_TIG_IZ_VA7E_argc
	.align 4
	.type	_TIG_IZ_VA7E_argc, @object
	.size	_TIG_IZ_VA7E_argc, 4
_TIG_IZ_VA7E_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Warm"
.LC1:
	.string	"Correct"
.LC2:
	.string	"Not a valid input"
.LC3:
	.string	"Cold"
.LC4:
	.string	"Not a vlaid input"
	.align 8
.LC5:
	.string	"Guess a whole number between -6 and 6: "
.LC6:
	.string	"%d"
	.align 8
.LC7:
	.string	"The statement on the next line is found using if and else statements"
.LC8:
	.string	"Hot"
	.align 8
.LC9:
	.string	"The statement on the next line is found using switch case statements"
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
	movq	$0, _TIG_IZ_VA7E_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_VA7E_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_VA7E_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 127 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-VA7E--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_VA7E_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_VA7E_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_VA7E_envp(%rip)
	nop
	movq	$31, -16(%rbp)
.L72:
	cmpq	$37, -16(%rbp)
	ja	.L75
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
	.long	.L40-.L8
	.long	.L39-.L8
	.long	.L38-.L8
	.long	.L37-.L8
	.long	.L36-.L8
	.long	.L35-.L8
	.long	.L34-.L8
	.long	.L33-.L8
	.long	.L32-.L8
	.long	.L75-.L8
	.long	.L31-.L8
	.long	.L30-.L8
	.long	.L29-.L8
	.long	.L28-.L8
	.long	.L27-.L8
	.long	.L75-.L8
	.long	.L26-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L75-.L8
	.long	.L75-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L75-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L24:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$17, -16(%rbp)
	jmp	.L41
.L19:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$20, -16(%rbp)
	jmp	.L41
.L36:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$17, -16(%rbp)
	jmp	.L41
.L14:
	movl	-20(%rbp), %eax
	cmpl	$5, %eax
	jne	.L42
	movq	$3, -16(%rbp)
	jmp	.L41
.L42:
	movq	$14, -16(%rbp)
	jmp	.L41
.L27:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$20, -16(%rbp)
	jmp	.L41
.L13:
	movq	$32, -16(%rbp)
	jmp	.L41
.L29:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$20, -16(%rbp)
	jmp	.L41
.L32:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$17, -16(%rbp)
	jmp	.L41
.L39:
	movl	-20(%rbp), %eax
	cmpl	$1, %eax
	jne	.L44
	movq	$10, -16(%rbp)
	jmp	.L41
.L44:
	movq	$7, -16(%rbp)
	jmp	.L41
.L21:
	movl	-20(%rbp), %eax
	cmpl	$-3, %eax
	jne	.L46
	movq	$25, -16(%rbp)
	jmp	.L41
.L46:
	movq	$16, -16(%rbp)
	jmp	.L41
.L37:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$20, -16(%rbp)
	jmp	.L41
.L26:
	movl	-20(%rbp), %eax
	cmpl	$-2, %eax
	jne	.L48
	movq	$28, -16(%rbp)
	jmp	.L41
.L48:
	movq	$36, -16(%rbp)
	jmp	.L41
.L20:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$20, -16(%rbp)
	jmp	.L41
.L9:
	movl	-20(%rbp), %eax
	cmpl	$2, %eax
	jne	.L50
	movq	$33, -16(%rbp)
	jmp	.L41
.L50:
	movq	$13, -16(%rbp)
	jmp	.L41
.L18:
	movl	-20(%rbp), %eax
	cmpl	$-5, %eax
	jne	.L52
	movq	$37, -16(%rbp)
	jmp	.L41
.L52:
	movq	$5, -16(%rbp)
	jmp	.L41
.L30:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$17, -16(%rbp)
	jmp	.L41
.L28:
	movl	-20(%rbp), %eax
	cmpl	$3, %eax
	jne	.L54
	movq	$35, -16(%rbp)
	jmp	.L41
.L54:
	movq	$26, -16(%rbp)
	jmp	.L41
.L23:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$20, -16(%rbp)
	jmp	.L41
.L12:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$6, -16(%rbp)
	jmp	.L41
.L25:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L73
	jmp	.L74
.L34:
	movl	-20(%rbp), %eax
	testl	%eax, %eax
	jne	.L57
	movq	$19, -16(%rbp)
	jmp	.L41
.L57:
	movq	$1, -16(%rbp)
	jmp	.L41
.L17:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$20, -16(%rbp)
	jmp	.L41
.L16:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$20, -16(%rbp)
	jmp	.L41
.L35:
	movl	-20(%rbp), %eax
	cmpl	$-4, %eax
	jne	.L59
	movq	$12, -16(%rbp)
	jmp	.L41
.L59:
	movq	$0, -16(%rbp)
	jmp	.L41
.L11:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$20, -16(%rbp)
	jmp	.L41
.L7:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$20, -16(%rbp)
	jmp	.L41
.L31:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$20, -16(%rbp)
	jmp	.L41
.L40:
	movl	-20(%rbp), %eax
	cmpl	$4, %eax
	jne	.L61
	movq	$24, -16(%rbp)
	jmp	.L41
.L61:
	movq	$30, -16(%rbp)
	jmp	.L41
.L33:
	movl	-20(%rbp), %eax
	cmpl	$-1, %eax
	jne	.L63
	movq	$27, -16(%rbp)
	jmp	.L41
.L63:
	movq	$23, -16(%rbp)
	jmp	.L41
.L10:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$20, -16(%rbp)
	jmp	.L41
.L15:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$17, -16(%rbp)
	jmp	.L41
.L38:
	movl	-20(%rbp), %eax
	addl	$5, %eax
	cmpl	$10, %eax
	ja	.L65
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L67(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L67(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L67:
	.long	.L66-.L67
	.long	.L66-.L67
	.long	.L68-.L67
	.long	.L68-.L67
	.long	.L69-.L67
	.long	.L70-.L67
	.long	.L69-.L67
	.long	.L68-.L67
	.long	.L68-.L67
	.long	.L66-.L67
	.long	.L66-.L67
	.text
.L66:
	movq	$8, -16(%rbp)
	jmp	.L71
.L68:
	movq	$18, -16(%rbp)
	jmp	.L71
.L69:
	movq	$29, -16(%rbp)
	jmp	.L71
.L70:
	movq	$4, -16(%rbp)
	jmp	.L71
.L65:
	movq	$11, -16(%rbp)
	nop
.L71:
	jmp	.L41
.L22:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -16(%rbp)
	jmp	.L41
.L75:
	nop
.L41:
	jmp	.L72
.L74:
	call	__stack_chk_fail@PLT
.L73:
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
