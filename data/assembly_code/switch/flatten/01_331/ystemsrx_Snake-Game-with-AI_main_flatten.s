	.file	"ystemsrx_Snake-Game-with-AI_main_flatten.c"
	.text
	.globl	_TIG_IZ_ecVH_argv
	.bss
	.align 8
	.type	_TIG_IZ_ecVH_argv, @object
	.size	_TIG_IZ_ecVH_argv, 8
_TIG_IZ_ecVH_argv:
	.zero	8
	.globl	_TIG_IZ_ecVH_argc
	.align 4
	.type	_TIG_IZ_ecVH_argc, @object
	.size	_TIG_IZ_ecVH_argc, 4
_TIG_IZ_ecVH_argc:
	.zero	4
	.globl	_TIG_IZ_ecVH_envp
	.align 8
	.type	_TIG_IZ_ecVH_envp, @object
	.size	_TIG_IZ_ecVH_envp, 8
_TIG_IZ_ecVH_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Invalid choice!"
.LC1:
	.string	"Easy"
.LC2:
	.string	"Select Difficulty:"
.LC3:
	.string	"1. Easy"
.LC4:
	.string	"2. Normal"
.LC5:
	.string	"3. Expert"
.LC6:
	.string	"4. Master"
.LC7:
	.string	"Enter your choice: "
.LC8:
	.string	"%d"
.LC9:
	.string	"Expert"
.LC10:
	.string	"Master"
.LC11:
	.string	"Normal"
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
	movq	$0, _TIG_IZ_ecVH_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_ecVH_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_ecVH_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 128 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-ecVH--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_ecVH_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_ecVH_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_ecVH_envp(%rip)
	nop
	movq	$4, -16(%rbp)
.L33:
	cmpq	$20, -16(%rbp)
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
	.long	.L36-.L8
	.long	.L36-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L36-.L8
	.long	.L17-.L8
	.long	.L36-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L36-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L36-.L8
	.long	.L36-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L36-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L10:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$20, -16(%rbp)
	jmp	.L21
.L18:
	movq	$3, -16(%rbp)
	jmp	.L21
.L12:
	movl	-24(%rbp), %eax
	cmpl	$4, %eax
	jg	.L22
	movq	$16, -16(%rbp)
	jmp	.L21
.L22:
	movq	$18, -16(%rbp)
	jmp	.L21
.L13:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$20, -16(%rbp)
	jmp	.L21
.L16:
	movl	$200, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	startGame
	movq	$20, -16(%rbp)
	jmp	.L21
.L19:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$9, -16(%rbp)
	jmp	.L21
.L11:
	movl	-24(%rbp), %eax
	cmpl	$4, %eax
	je	.L24
	cmpl	$4, %eax
	jg	.L25
	cmpl	$3, %eax
	je	.L26
	cmpl	$3, %eax
	jg	.L25
	cmpl	$1, %eax
	je	.L27
	cmpl	$2, %eax
	je	.L28
	jmp	.L25
.L24:
	movq	$19, -16(%rbp)
	jmp	.L29
.L26:
	movq	$11, -16(%rbp)
	jmp	.L29
.L28:
	movq	$2, -16(%rbp)
	jmp	.L29
.L27:
	movq	$8, -16(%rbp)
	jmp	.L29
.L25:
	movq	$6, -16(%rbp)
	nop
.L29:
	jmp	.L21
.L14:
	movl	$100, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	startGame
	movq	$20, -16(%rbp)
	jmp	.L21
.L15:
	movl	-24(%rbp), %eax
	testl	%eax, %eax
	jle	.L30
	movq	$15, -16(%rbp)
	jmp	.L21
.L30:
	movq	$12, -16(%rbp)
	jmp	.L21
.L9:
	movl	$80, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	startGame
	movq	$20, -16(%rbp)
	jmp	.L21
.L17:
	movq	$20, -16(%rbp)
	jmp	.L21
.L20:
	movl	$120, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	startGame
	movq	$20, -16(%rbp)
	jmp	.L21
.L7:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L34
	jmp	.L35
.L36:
	nop
.L21:
	jmp	.L33
.L35:
	call	__stack_chk_fail@PLT
.L34:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.section	.rodata
.LC12:
	.string	"gcc -o %s.exe %s.c"
.LC13:
	.string	"%s.exe %d"
	.text
	.globl	compileAndRun
	.type	compileAndRun, @function
compileAndRun:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$560, %rsp
	movq	%rdi, -552(%rbp)
	movl	%esi, -556(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$1, -536(%rbp)
.L43:
	cmpq	$2, -536(%rbp)
	je	.L38
	cmpq	$2, -536(%rbp)
	ja	.L46
	cmpq	$0, -536(%rbp)
	je	.L47
	cmpq	$1, -536(%rbp)
	jne	.L46
	movq	$2, -536(%rbp)
	jmp	.L41
.L38:
	movq	-552(%rbp), %rcx
	movq	-552(%rbp), %rdx
	leaq	-528(%rbp), %rax
	movq	%rcx, %r8
	movq	%rdx, %rcx
	leaq	.LC12(%rip), %rdx
	movl	$256, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-528(%rbp), %rax
	movq	%rax, %rdi
	call	system@PLT
	movl	-556(%rbp), %ecx
	movq	-552(%rbp), %rdx
	leaq	-272(%rbp), %rax
	movl	%ecx, %r8d
	movq	%rdx, %rcx
	leaq	.LC13(%rip), %rdx
	movl	$256, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-272(%rbp), %rax
	movq	%rax, %rdi
	call	system@PLT
	movq	$0, -536(%rbp)
	jmp	.L41
.L46:
	nop
.L41:
	jmp	.L43
.L47:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L45
	call	__stack_chk_fail@PLT
.L45:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	compileAndRun, .-compileAndRun
	.section	.rodata
	.align 8
.LC14:
	.string	"Starting game %s at speed %d...\n"
	.text
	.globl	startGame
	.type	startGame, @function
startGame:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$0, -8(%rbp)
.L54:
	cmpq	$2, -8(%rbp)
	je	.L55
	cmpq	$2, -8(%rbp)
	ja	.L56
	cmpq	$0, -8(%rbp)
	je	.L51
	cmpq	$1, -8(%rbp)
	jne	.L56
	movl	-28(%rbp), %edx
	movq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-28(%rbp), %edx
	movq	-24(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	compileAndRun
	movq	$2, -8(%rbp)
	jmp	.L52
.L51:
	movq	$1, -8(%rbp)
	jmp	.L52
.L56:
	nop
.L52:
	jmp	.L54
.L55:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	startGame, .-startGame
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
