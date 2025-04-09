	.file	"mn-48_CP-Assignment_C_flatten.c"
	.text
	.globl	_TIG_IZ_CrGS_argv
	.bss
	.align 8
	.type	_TIG_IZ_CrGS_argv, @object
	.size	_TIG_IZ_CrGS_argv, 8
_TIG_IZ_CrGS_argv:
	.zero	8
	.globl	_TIG_IZ_CrGS_envp
	.align 8
	.type	_TIG_IZ_CrGS_envp, @object
	.size	_TIG_IZ_CrGS_envp, 8
_TIG_IZ_CrGS_envp:
	.zero	8
	.globl	_TIG_IZ_CrGS_argc
	.align 4
	.type	_TIG_IZ_CrGS_argc, @object
	.size	_TIG_IZ_CrGS_argc, 4
_TIG_IZ_CrGS_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"Invalid range. Ensure A < B."
.LC1:
	.string	"Low"
	.align 8
.LC2:
	.string	"Sorry, you ran out of tries. The number was %d\n"
.LC3:
	.string	"High"
	.align 8
.LC4:
	.string	"Congratulations! You guessed the number!"
.LC5:
	.string	"Your Score = %d\n"
.LC6:
	.string	"Guess: "
.LC7:
	.string	"%d"
	.text
	.globl	guessNumberGame
	.type	guessNumberGame, @function
guessNumberGame:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -52(%rbp)
	movl	%esi, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$8, -24(%rbp)
.L28:
	movq	-24(%rbp), %rax
	subq	$3, %rax
	cmpq	$15, %rax
	ja	.L31
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
	.long	.L32-.L4
	.long	.L32-.L4
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L32-.L4
	.long	.L31-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L31-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L3:
	movl	$0, %edi
	call	time@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movl	%eax, %edi
	call	srand@PLT
	call	rand@PLT
	movl	%eax, -28(%rbp)
	movl	-56(%rbp), %eax
	subl	-52(%rbp), %eax
	leal	1(%rax), %ecx
	movl	-28(%rbp), %eax
	cltd
	idivl	%ecx
	movl	-52(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -40(%rbp)
	movl	$0, -36(%rbp)
	movl	$10, -32(%rbp)
	movq	$5, -24(%rbp)
	jmp	.L18
.L7:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$3, -24(%rbp)
	jmp	.L18
.L9:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -24(%rbp)
	jmp	.L18
.L12:
	movl	-52(%rbp), %eax
	cmpl	-56(%rbp), %eax
	jl	.L20
	movq	$14, -24(%rbp)
	jmp	.L18
.L20:
	movq	$18, -24(%rbp)
	jmp	.L18
.L6:
	movl	-40(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$4, -24(%rbp)
	jmp	.L18
.L10:
	movl	-44(%rbp), %eax
	cmpl	%eax, -40(%rbp)
	jne	.L22
	movq	$6, -24(%rbp)
	jmp	.L18
.L22:
	movq	$17, -24(%rbp)
	jmp	.L18
.L8:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -24(%rbp)
	jmp	.L18
.L5:
	movl	-44(%rbp), %eax
	cmpl	%eax, -40(%rbp)
	jle	.L24
	movq	$12, -24(%rbp)
	jmp	.L18
.L24:
	movq	$13, -24(%rbp)
	jmp	.L18
.L14:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$11, %eax
	subl	-36(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$9, -24(%rbp)
	jmp	.L18
.L15:
	movl	-36(%rbp), %eax
	cmpl	-32(%rbp), %eax
	jge	.L26
	movq	$7, -24(%rbp)
	jmp	.L18
.L26:
	movq	$16, -24(%rbp)
	jmp	.L18
.L13:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-44(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -36(%rbp)
	movq	$11, -24(%rbp)
	jmp	.L18
.L31:
	nop
.L18:
	jmp	.L28
.L32:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L30
	call	__stack_chk_fail@PLT
.L30:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	guessNumberGame, .-guessNumberGame
	.section	.rodata
.LC8:
	.string	"Input:"
.LC9:
	.string	"A = "
.LC10:
	.string	"B = "
.LC11:
	.string	"\nOutput:"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_CrGS_envp(%rip)
	nop
.L34:
	movq	$0, _TIG_IZ_CrGS_argv(%rip)
	nop
.L35:
	movl	$0, _TIG_IZ_CrGS_argc(%rip)
	nop
	nop
.L36:
.L37:
#APP
# 104 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-CrGS--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_CrGS_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_CrGS_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_CrGS_envp(%rip)
	nop
	movq	$0, -16(%rbp)
.L43:
	cmpq	$2, -16(%rbp)
	je	.L38
	cmpq	$2, -16(%rbp)
	ja	.L46
	cmpq	$0, -16(%rbp)
	je	.L40
	cmpq	$1, -16(%rbp)
	jne	.L46
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-20(%rbp), %edx
	movl	-24(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	guessNumberGame
	movq	$2, -16(%rbp)
	jmp	.L41
.L40:
	movq	$1, -16(%rbp)
	jmp	.L41
.L38:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L44
	jmp	.L45
.L46:
	nop
.L41:
	jmp	.L43
.L45:
	call	__stack_chk_fail@PLT
.L44:
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
