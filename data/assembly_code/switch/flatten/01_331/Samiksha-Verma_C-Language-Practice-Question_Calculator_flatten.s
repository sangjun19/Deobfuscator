	.file	"Samiksha-Verma_C-Language-Practice-Question_Calculator_flatten.c"
	.text
	.globl	_TIG_IZ_XG0A_argc
	.bss
	.align 4
	.type	_TIG_IZ_XG0A_argc, @object
	.size	_TIG_IZ_XG0A_argc, 4
_TIG_IZ_XG0A_argc:
	.zero	4
	.globl	_TIG_IZ_XG0A_argv
	.align 8
	.type	_TIG_IZ_XG0A_argv, @object
	.size	_TIG_IZ_XG0A_argv, 8
_TIG_IZ_XG0A_argv:
	.zero	8
	.globl	_TIG_IZ_XG0A_envp
	.align 8
	.type	_TIG_IZ_XG0A_envp, @object
	.size	_TIG_IZ_XG0A_envp, 8
_TIG_IZ_XG0A_envp:
	.zero	8
	.text
	.globl	multiply
	.type	multiply, @function
multiply:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L4:
	cmpq	$0, -8(%rbp)
	jne	.L7
	movl	-20(%rbp), %eax
	imull	-24(%rbp), %eax
	jmp	.L6
.L7:
	nop
	jmp	.L4
.L6:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	multiply, .-multiply
	.globl	subtract
	.type	subtract, @function
subtract:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L11:
	cmpq	$0, -8(%rbp)
	jne	.L14
	movl	-20(%rbp), %eax
	subl	-24(%rbp), %eax
	jmp	.L13
.L14:
	nop
	jmp	.L11
.L13:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	subtract, .-subtract
	.section	.rodata
.LC0:
	.string	"Result: %d\n"
.LC1:
	.string	"Invalid choice."
.LC2:
	.string	"Enter two numbers: "
.LC3:
	.string	"%d %d"
.LC4:
	.string	"Enter your choice:"
.LC5:
	.string	"1. Addition"
.LC6:
	.string	"2. Subtraction"
.LC7:
	.string	"3. Multiplication"
.LC8:
	.string	"4. Division"
.LC9:
	.string	"%d"
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
	movq	$0, _TIG_IZ_XG0A_envp(%rip)
	nop
.L16:
	movq	$0, _TIG_IZ_XG0A_argv(%rip)
	nop
.L17:
	movl	$0, _TIG_IZ_XG0A_argc(%rip)
	nop
	nop
.L18:
.L19:
#APP
# 168 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-XG0A--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_XG0A_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_XG0A_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_XG0A_envp(%rip)
	nop
	movq	$5, -16(%rbp)
.L41:
	cmpq	$14, -16(%rbp)
	ja	.L44
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L22(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L22(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L22:
	.long	.L32-.L22
	.long	.L31-.L22
	.long	.L30-.L22
	.long	.L29-.L22
	.long	.L28-.L22
	.long	.L27-.L22
	.long	.L26-.L22
	.long	.L44-.L22
	.long	.L25-.L22
	.long	.L44-.L22
	.long	.L44-.L22
	.long	.L44-.L22
	.long	.L24-.L22
	.long	.L23-.L22
	.long	.L21-.L22
	.text
.L28:
	movl	-28(%rbp), %edx
	movl	-32(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	multiply
	movl	%eax, -20(%rbp)
	movq	$8, -16(%rbp)
	jmp	.L33
.L21:
	movl	$0, %eax
	jmp	.L42
.L24:
	movl	$1, %eax
	jmp	.L42
.L25:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$14, -16(%rbp)
	jmp	.L33
.L31:
	movl	-28(%rbp), %edx
	movl	-32(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	add
	movl	%eax, -20(%rbp)
	movq	$8, -16(%rbp)
	jmp	.L33
.L29:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$12, -16(%rbp)
	jmp	.L33
.L23:
	movl	-28(%rbp), %edx
	movl	-32(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	divide
	movl	%eax, -20(%rbp)
	movq	$8, -16(%rbp)
	jmp	.L33
.L26:
	movl	-28(%rbp), %edx
	movl	-32(%rbp), %eax
	movl	%edx, %esi
	movl	%eax, %edi
	call	subtract
	movl	%eax, -20(%rbp)
	movq	$8, -16(%rbp)
	jmp	.L33
.L27:
	movq	$0, -16(%rbp)
	jmp	.L33
.L32:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-28(%rbp), %rdx
	leaq	-32(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
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
	call	puts@PLT
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$2, -16(%rbp)
	jmp	.L33
.L30:
	movl	-24(%rbp), %eax
	cmpl	$4, %eax
	je	.L35
	cmpl	$4, %eax
	jg	.L36
	cmpl	$3, %eax
	je	.L37
	cmpl	$3, %eax
	jg	.L36
	cmpl	$1, %eax
	je	.L38
	cmpl	$2, %eax
	je	.L39
	jmp	.L36
.L35:
	movq	$13, -16(%rbp)
	jmp	.L40
.L37:
	movq	$4, -16(%rbp)
	jmp	.L40
.L39:
	movq	$6, -16(%rbp)
	jmp	.L40
.L38:
	movq	$1, -16(%rbp)
	jmp	.L40
.L36:
	movq	$3, -16(%rbp)
	nop
.L40:
	jmp	.L33
.L44:
	nop
.L33:
	jmp	.L41
.L42:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L43
	call	__stack_chk_fail@PLT
.L43:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.section	.rodata
.LC10:
	.string	"Error! Division by zero."
	.text
	.globl	divide
	.type	divide, @function
divide:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L55:
	cmpq	$4, -8(%rbp)
	je	.L46
	cmpq	$4, -8(%rbp)
	ja	.L56
	cmpq	$3, -8(%rbp)
	je	.L48
	cmpq	$3, -8(%rbp)
	ja	.L56
	cmpq	$0, -8(%rbp)
	je	.L49
	cmpq	$2, -8(%rbp)
	je	.L50
	jmp	.L56
.L46:
	movl	$0, %eax
	jmp	.L51
.L48:
	movl	-20(%rbp), %eax
	cltd
	idivl	-24(%rbp)
	jmp	.L51
.L49:
	cmpl	$0, -24(%rbp)
	jne	.L52
	movq	$2, -8(%rbp)
	jmp	.L54
.L52:
	movq	$3, -8(%rbp)
	jmp	.L54
.L50:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$4, -8(%rbp)
	jmp	.L54
.L56:
	nop
.L54:
	jmp	.L55
.L51:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	divide, .-divide
	.globl	add
	.type	add, @function
add:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, -20(%rbp)
	movl	%esi, -24(%rbp)
	movq	$0, -8(%rbp)
.L60:
	cmpq	$0, -8(%rbp)
	jne	.L63
	movl	-20(%rbp), %edx
	movl	-24(%rbp), %eax
	addl	%edx, %eax
	jmp	.L62
.L63:
	nop
	jmp	.L60
.L62:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	add, .-add
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
