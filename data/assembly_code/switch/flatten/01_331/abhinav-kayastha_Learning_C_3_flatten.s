	.file	"abhinav-kayastha_Learning_C_3_flatten.c"
	.text
	.globl	_TIG_IZ_ebXo_argv
	.bss
	.align 8
	.type	_TIG_IZ_ebXo_argv, @object
	.size	_TIG_IZ_ebXo_argv, 8
_TIG_IZ_ebXo_argv:
	.zero	8
	.globl	_TIG_IZ_ebXo_envp
	.align 8
	.type	_TIG_IZ_ebXo_envp, @object
	.size	_TIG_IZ_ebXo_envp, 8
_TIG_IZ_ebXo_envp:
	.zero	8
	.globl	_TIG_IZ_ebXo_argc
	.align 4
	.type	_TIG_IZ_ebXo_argc, @object
	.size	_TIG_IZ_ebXo_argc, 4
_TIG_IZ_ebXo_argc:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"Enter the grade for student %d (0 - 5 or -1 to cancel): "
.LC1:
	.string	"%d"
	.align 8
.LC2:
	.string	"Exiting student information input."
.LC3:
	.string	"Student\tGrade"
	.align 8
.LC4:
	.string	"Invalid student number. Try again."
	.align 8
.LC5:
	.string	"Invalid grade. Please enter a grade between 0 and 5 or -1."
.LC6:
	.string	"N/A"
.LC7:
	.string	"%d\t"
.LC8:
	.string	"%d\n"
	.align 8
.LC9:
	.string	"Enter the number of students: "
	.align 8
.LC10:
	.string	"Enter student number (1 - %d) or 0 to stop: "
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
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_ebXo_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_ebXo_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_ebXo_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 113 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-ebXo--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_ebXo_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_ebXo_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_ebXo_envp(%rip)
	nop
	movq	$28, -24(%rbp)
.L53:
	cmpq	$37, -24(%rbp)
	ja	.L55
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
	.long	.L55-.L8
	.long	.L55-.L8
	.long	.L31-.L8
	.long	.L55-.L8
	.long	.L30-.L8
	.long	.L29-.L8
	.long	.L28-.L8
	.long	.L55-.L8
	.long	.L27-.L8
	.long	.L55-.L8
	.long	.L26-.L8
	.long	.L25-.L8
	.long	.L55-.L8
	.long	.L55-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L55-.L8
	.long	.L22-.L8
	.long	.L21-.L8
	.long	.L20-.L8
	.long	.L55-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L16-.L8
	.long	.L15-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L55-.L8
	.long	.L55-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L55-.L8
	.long	.L55-.L8
	.long	.L9-.L8
	.long	.L7-.L8
	.text
.L22:
	movl	-52(%rbp), %eax
	cmpl	%eax, -40(%rbp)
	jge	.L33
	movq	$20, -24(%rbp)
	jmp	.L35
.L33:
	movq	$7, -24(%rbp)
	jmp	.L35
.L16:
	movl	-48(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-44(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$37, -24(%rbp)
	jmp	.L35
.L24:
	movl	-52(%rbp), %eax
	cmpl	%eax, -36(%rbp)
	jge	.L36
	movq	$9, -24(%rbp)
	jmp	.L35
.L36:
	movq	$7, -24(%rbp)
	jmp	.L35
.L12:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -36(%rbp)
	movq	$15, -24(%rbp)
	jmp	.L35
.L25:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -24(%rbp)
	jmp	.L35
.L18:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -24(%rbp)
	jmp	.L35
.L31:
	movl	-44(%rbp), %eax
	cmpl	$5, %eax
	jg	.L38
	movq	$11, -24(%rbp)
	jmp	.L35
.L38:
	movq	$0, -24(%rbp)
	jmp	.L35
.L23:
	movl	-48(%rbp), %eax
	testl	%eax, %eax
	jne	.L40
	movq	$31, -24(%rbp)
	jmp	.L41
.L40:
	movq	$27, -24(%rbp)
	nop
.L41:
	jmp	.L35
.L17:
	addl	$1, -36(%rbp)
	movq	$15, -24(%rbp)
	jmp	.L35
.L9:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -24(%rbp)
	jmp	.L35
.L15:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$24, -24(%rbp)
	jmp	.L35
.L26:
	movl	-48(%rbp), %eax
	subl	$1, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rax, %rdx
	movl	-44(%rbp), %eax
	movl	%eax, (%rdx)
	movq	$7, -24(%rbp)
	jmp	.L35
.L27:
	movl	-36(%rbp), %eax
	addl	$1, %eax
	movl	%eax, %esi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$5, -24(%rbp)
	jmp	.L35
.L21:
	movl	-52(%rbp), %eax
	cltq
	salq	$5, %rax
	addq	$31, %rax
	shrq	$3, %rax
	movq	%rax, %rdx
	movabsq	$2305843009213693948, %rax
	andq	%rdx, %rax
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	leaq	8(%rax), %rdx
	movl	$16, %eax
	subq	$1, %rax
	addq	%rdx, %rax
	movl	$16, %ecx
	movl	$0, %edx
	divq	%rcx
	imulq	$16, %rax, %rax
	movq	%rax, %rcx
	andq	$-4096, %rcx
	movq	%rsp, %rdx
	subq	%rcx, %rdx
.L42:
	cmpq	%rdx, %rsp
	je	.L43
	subq	$4096, %rsp
	orq	$0, 4088(%rsp)
	jmp	.L42
.L43:
	movq	%rax, %rdx
	andl	$4095, %edx
	subq	%rdx, %rsp
	movq	%rax, %rdx
	andl	$4095, %edx
	testq	%rdx, %rdx
	je	.L44
	andl	$4095, %eax
	subq	$8, %rax
	addq	%rsp, %rax
	orq	$0, (%rax)
.L44:
	movq	%rsp, %rax
	addq	$15, %rax
	shrq	$4, %rax
	salq	$4, %rax
	movq	%rax, -32(%rbp)
	movq	$6, -24(%rbp)
	jmp	.L35
.L11:
	movl	-48(%rbp), %edx
	movl	-52(%rbp), %eax
	cmpl	%eax, %edx
	jle	.L45
	movq	$12, -24(%rbp)
	jmp	.L35
.L45:
	movq	$25, -24(%rbp)
	jmp	.L35
.L29:
	movl	$0, -40(%rbp)
	movq	$18, -24(%rbp)
	jmp	.L35
.L14:
	movl	-48(%rbp), %eax
	testl	%eax, %eax
	jg	.L47
	movq	$36, -24(%rbp)
	jmp	.L35
.L47:
	movq	$32, -24(%rbp)
	jmp	.L35
.L19:
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$24, -24(%rbp)
	jmp	.L35
.L13:
	movq	$33, -24(%rbp)
	jmp	.L35
.L30:
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	$-1, %eax
	jne	.L49
	movq	$26, -24(%rbp)
	jmp	.L35
.L49:
	movq	$22, -24(%rbp)
	jmp	.L35
.L10:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-52(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$19, -24(%rbp)
	jmp	.L35
.L7:
	movl	-44(%rbp), %eax
	cmpl	$-1, %eax
	jl	.L51
	movq	$3, -24(%rbp)
	jmp	.L35
.L51:
	movq	$23, -24(%rbp)
	jmp	.L35
.L32:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$7, -24(%rbp)
	jmp	.L35
.L28:
	movl	-52(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$16, -24(%rbp)
	jmp	.L35
.L20:
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movl	$-1, (%rax)
	addl	$1, -40(%rbp)
	movq	$18, -24(%rbp)
	jmp	.L35
.L55:
	nop
.L35:
	jmp	.L53
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
