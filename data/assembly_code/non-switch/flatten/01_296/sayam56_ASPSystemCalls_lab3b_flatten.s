	.file	"sayam56_ASPSystemCalls_lab3b_flatten.c"
	.text
	.globl	_TIG_IZ_3sGB_envp
	.bss
	.align 8
	.type	_TIG_IZ_3sGB_envp, @object
	.size	_TIG_IZ_3sGB_envp, 8
_TIG_IZ_3sGB_envp:
	.zero	8
	.globl	_TIG_IZ_3sGB_argc
	.align 4
	.type	_TIG_IZ_3sGB_argc, @object
	.size	_TIG_IZ_3sGB_argc, 4
_TIG_IZ_3sGB_argc:
	.zero	4
	.globl	_TIG_IZ_3sGB_argv
	.align 8
	.type	_TIG_IZ_3sGB_argv, @object
	.size	_TIG_IZ_3sGB_argv, 8
_TIG_IZ_3sGB_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"\nPID for child : %d\n"
.LC1:
	.string	"PPID for child: %d\n"
.LC2:
	.string	"\n pid of parent process: %d\n"
.LC3:
	.string	"Enter an integer 1 2 or 3: "
.LC4:
	.string	"%d"
.LC5:
	.string	"Number = %d"
	.align 8
.LC6:
	.string	"Child process terminated with status: %d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
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
	movq	$0, _TIG_IZ_3sGB_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_3sGB_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_3sGB_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 157 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-3sGB--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_3sGB_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_3sGB_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_3sGB_envp(%rip)
	nop
	movq	$5, -16(%rbp)
.L39:
	cmpq	$22, -16(%rbp)
	ja	.L42
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
	.long	.L24-.L8
	.long	.L42-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L42-.L8
	.long	.L19-.L8
	.long	.L42-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L42-.L8
	.long	.L10-.L8
	.long	.L42-.L8
	.long	.L9-.L8
	.long	.L42-.L8
	.long	.L7-.L8
	.text
.L10:
	call	getpid@PLT
	movl	%eax, -56(%rbp)
	movl	-56(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	call	getppid@PLT
	movl	%eax, -52(%rbp)
	movl	-52(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, %edi
	call	sleep@PLT
	addl	$1, -64(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L25
.L21:
	leaq	-68(%rbp), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	wait@PLT
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, -24(%rbp)
	call	getpid@PLT
	movl	%eax, -20(%rbp)
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$15, -16(%rbp)
	jmp	.L25
.L13:
	movl	-72(%rbp), %eax
	cmpl	$3, %eax
	jne	.L26
	movq	$3, -16(%rbp)
	jmp	.L25
.L26:
	movq	$4, -16(%rbp)
	jmp	.L25
.L12:
	movl	-68(%rbp), %eax
	andl	$127, %eax
	testl	%eax, %eax
	jne	.L28
	movq	$13, -16(%rbp)
	jmp	.L25
.L28:
	movq	$16, -16(%rbp)
	jmp	.L25
.L15:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-72(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-72(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	call	fork@PLT
	movl	%eax, -48(%rbp)
	movq	$22, -16(%rbp)
	jmp	.L25
.L22:
	movl	$0, -64(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L25
.L11:
	movl	-68(%rbp), %eax
	andl	$127, %eax
	addl	$1, %eax
	sarb	%al
	testb	%al, %al
	jle	.L30
	movq	$2, -16(%rbp)
	jmp	.L25
.L30:
	movq	$11, -16(%rbp)
	jmp	.L25
.L16:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L40
	jmp	.L41
.L18:
	call	getpid@PLT
	movl	%eax, -44(%rbp)
	movl	-44(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	call	getppid@PLT
	movl	%eax, -40(%rbp)
	movl	-40(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$7, %edi
	call	exit@PLT
.L14:
	movl	-68(%rbp), %eax
	sarl	$8, %eax
	movzbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$16, -16(%rbp)
	jmp	.L25
.L7:
	movl	-72(%rbp), %eax
	cmpl	$1, %eax
	jne	.L33
	movq	$7, -16(%rbp)
	jmp	.L25
.L33:
	movq	$0, -16(%rbp)
	jmp	.L25
.L20:
	movq	$12, -16(%rbp)
	jmp	.L25
.L17:
	cmpl	$2, -64(%rbp)
	jg	.L35
	movq	$18, -16(%rbp)
	jmp	.L25
.L35:
	movq	$20, -16(%rbp)
	jmp	.L25
.L24:
	movl	-72(%rbp), %eax
	cmpl	$2, %eax
	jne	.L37
	movq	$9, -16(%rbp)
	jmp	.L25
.L37:
	movq	$14, -16(%rbp)
	jmp	.L25
.L19:
	call	getpid@PLT
	movl	%eax, -36(%rbp)
	movl	-36(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	call	getppid@PLT
	movl	%eax, -32(%rbp)
	movl	-32(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -16(%rbp)
	jmp	.L25
.L23:
	movl	-68(%rbp), %eax
	andl	$127, %eax
	movl	%eax, %esi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$11, -16(%rbp)
	jmp	.L25
.L9:
	movl	$100, %eax
	movl	$0, %ecx
	cltd
	idivl	%ecx
	movl	%eax, -60(%rbp)
	movq	$11, -16(%rbp)
	jmp	.L25
.L42:
	nop
.L25:
	jmp	.L39
.L41:
	call	__stack_chk_fail@PLT
.L40:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
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
