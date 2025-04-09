	.file	"mblevu_justChere_fizzbuzz_flatten.c"
	.text
	.globl	_TIG_IZ_MAf7_argv
	.bss
	.align 8
	.type	_TIG_IZ_MAf7_argv, @object
	.size	_TIG_IZ_MAf7_argv, 8
_TIG_IZ_MAf7_argv:
	.zero	8
	.globl	_TIG_IZ_MAf7_envp
	.align 8
	.type	_TIG_IZ_MAf7_envp, @object
	.size	_TIG_IZ_MAf7_envp, 8
_TIG_IZ_MAf7_envp:
	.zero	8
	.globl	_TIG_IZ_MAf7_argc
	.align 4
	.type	_TIG_IZ_MAf7_argc, @object
	.size	_TIG_IZ_MAf7_argc, 4
_TIG_IZ_MAf7_argc:
	.zero	4
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
	subq	$80, %rsp
	movl	%edi, -52(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_MAf7_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_MAf7_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_MAf7_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 130 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-MAf7--0
# 0 "" 2
#NO_APP
	movl	-52(%rbp), %eax
	movl	%eax, _TIG_IZ_MAf7_argc(%rip)
	movq	-64(%rbp), %rax
	movq	%rax, _TIG_IZ_MAf7_argv(%rip)
	movq	-72(%rbp), %rax
	movq	%rax, _TIG_IZ_MAf7_envp(%rip)
	nop
	movq	$1, -24(%rbp)
.L17:
	cmpq	$6, -24(%rbp)
	ja	.L20
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
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L20-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L20-.L8
	.long	.L7-.L8
	.text
.L9:
	movl	$2, -40(%rbp)
	movl	-40(%rbp), %eax
	leaq	-40(%rbp), %rdx
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	fizzBuzz
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -32(%rbp)
	movl	$0, -36(%rbp)
	movq	$3, -24(%rbp)
	jmp	.L13
.L11:
	movq	$4, -24(%rbp)
	jmp	.L13
.L10:
	movl	-40(%rbp), %eax
	cmpl	%eax, -36(%rbp)
	jge	.L14
	movq	$0, -24(%rbp)
	jmp	.L13
.L14:
	movq	$6, -24(%rbp)
	jmp	.L13
.L7:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L18
	jmp	.L19
.L12:
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	puts@PLT
	addl	$1, -36(%rbp)
	movq	$3, -24(%rbp)
	jmp	.L13
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
.LC0:
	.string	"%d"
.LC1:
	.string	"Fizz"
.LC2:
	.string	"Buzz"
.LC3:
	.string	"FizzBuzz"
	.text
	.globl	fizzBuzz
	.type	fizzBuzz, @function
fizzBuzz:
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
	movq	%rsi, -64(%rbp)
	movq	$17, -24(%rbp)
.L50:
	cmpq	$17, -24(%rbp)
	ja	.L52
	movq	-24(%rbp), %rax
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
	.long	.L37-.L24
	.long	.L52-.L24
	.long	.L36-.L24
	.long	.L35-.L24
	.long	.L34-.L24
	.long	.L52-.L24
	.long	.L52-.L24
	.long	.L33-.L24
	.long	.L32-.L24
	.long	.L31-.L24
	.long	.L30-.L24
	.long	.L29-.L24
	.long	.L28-.L24
	.long	.L27-.L24
	.long	.L26-.L24
	.long	.L52-.L24
	.long	.L25-.L24
	.long	.L23-.L24
	.text
.L34:
	movl	$12, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movl	-36(%rbp), %eax
	cltq
	salq	$3, %rax
	leaq	-8(%rax), %rdx
	movq	-32(%rbp), %rax
	addq	%rax, %rdx
	movq	-8(%rbp), %rax
	movq	%rax, (%rdx)
	movl	-36(%rbp), %eax
	cltq
	salq	$3, %rax
	leaq	-8(%rax), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movl	-36(%rbp), %edx
	leaq	.LC0(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	sprintf@PLT
	movq	$3, -24(%rbp)
	jmp	.L38
.L26:
	movl	-52(%rbp), %eax
	cltq
	salq	$3, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -32(%rbp)
	movl	$1, -36(%rbp)
	movq	$0, -24(%rbp)
	jmp	.L38
.L28:
	movl	-36(%rbp), %edx
	movslq	%edx, %rax
	imulq	$1717986919, %rax, %rax
	shrq	$32, %rax
	sarl	%eax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	movl	%eax, %ecx
	sall	$2, %ecx
	addl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	testl	%eax, %eax
	jne	.L39
	movq	$13, -24(%rbp)
	jmp	.L38
.L39:
	movq	$4, -24(%rbp)
	jmp	.L38
.L32:
	movl	-36(%rbp), %edx
	movslq	%edx, %rax
	imulq	$1431655766, %rax, %rax
	shrq	$32, %rax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	movl	%eax, %ecx
	addl	%ecx, %ecx
	addl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	testl	%eax, %eax
	jne	.L41
	movq	$16, -24(%rbp)
	jmp	.L38
.L41:
	movq	$12, -24(%rbp)
	jmp	.L38
.L35:
	addl	$1, -36(%rbp)
	movq	$0, -24(%rbp)
	jmp	.L38
.L25:
	movl	-36(%rbp), %eax
	cltq
	salq	$3, %rax
	leaq	-8(%rax), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	leaq	.LC1(%rip), %rdx
	movq	%rdx, (%rax)
	movq	$3, -24(%rbp)
	jmp	.L38
.L29:
	movq	-64(%rbp), %rax
	movl	-52(%rbp), %edx
	movl	%edx, (%rax)
	movq	$9, -24(%rbp)
	jmp	.L38
.L31:
	movq	-32(%rbp), %rax
	jmp	.L51
.L27:
	movl	-36(%rbp), %eax
	cltq
	salq	$3, %rax
	leaq	-8(%rax), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	leaq	.LC2(%rip), %rdx
	movq	%rdx, (%rax)
	movq	$3, -24(%rbp)
	jmp	.L38
.L23:
	movq	$14, -24(%rbp)
	jmp	.L38
.L30:
	movl	-36(%rbp), %edx
	movslq	%edx, %rax
	imulq	$1717986919, %rax, %rax
	shrq	$32, %rax
	sarl	%eax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	movl	%eax, %ecx
	sall	$2, %ecx
	addl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	testl	%eax, %eax
	jne	.L44
	movq	$2, -24(%rbp)
	jmp	.L38
.L44:
	movq	$8, -24(%rbp)
	jmp	.L38
.L37:
	movl	-36(%rbp), %eax
	cmpl	-52(%rbp), %eax
	jg	.L46
	movq	$7, -24(%rbp)
	jmp	.L38
.L46:
	movq	$11, -24(%rbp)
	jmp	.L38
.L33:
	movl	-36(%rbp), %edx
	movslq	%edx, %rax
	imulq	$1431655766, %rax, %rax
	shrq	$32, %rax
	movl	%edx, %ecx
	sarl	$31, %ecx
	subl	%ecx, %eax
	movl	%eax, %ecx
	addl	%ecx, %ecx
	addl	%eax, %ecx
	movl	%edx, %eax
	subl	%ecx, %eax
	testl	%eax, %eax
	jne	.L48
	movq	$10, -24(%rbp)
	jmp	.L38
.L48:
	movq	$8, -24(%rbp)
	jmp	.L38
.L36:
	movl	-36(%rbp), %eax
	cltq
	salq	$3, %rax
	leaq	-8(%rax), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	leaq	.LC3(%rip), %rdx
	movq	%rdx, (%rax)
	movq	$3, -24(%rbp)
	jmp	.L38
.L52:
	nop
.L38:
	jmp	.L50
.L51:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	fizzBuzz, .-fizzBuzz
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
