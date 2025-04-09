	.file	"halakundi_DSA-Program_1_flatten.c"
	.text
	.globl	_TIG_IZ_VfJR_argv
	.bss
	.align 8
	.type	_TIG_IZ_VfJR_argv, @object
	.size	_TIG_IZ_VfJR_argv, 8
_TIG_IZ_VfJR_argv:
	.zero	8
	.globl	_TIG_IZ_VfJR_envp
	.align 8
	.type	_TIG_IZ_VfJR_envp, @object
	.size	_TIG_IZ_VfJR_envp, 8
_TIG_IZ_VfJR_envp:
	.zero	8
	.globl	_TIG_IZ_VfJR_argc
	.align 4
	.type	_TIG_IZ_VfJR_argc, @object
	.size	_TIG_IZ_VfJR_argc, 4
_TIG_IZ_VfJR_argc:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"enter position (0 to %d)to delete :\t"
.LC1:
	.string	"%d"
	.align 8
.LC2:
	.string	"\n enter element and pos(0 to %d)to insert :\t"
.LC3:
	.string	"%d%d"
.LC4:
	.string	"%d \t"
	.align 8
.LC5:
	.string	"\n 1.insert \n 2.delete \n 3.display \n 4.exit \n enter your choice:\t"
.LC6:
	.string	"\n array elements are:"
	.align 8
.LC7:
	.string	"enter number of elements to create an array:\t"
.LC8:
	.string	"dynamic array created "
.LC9:
	.string	"enter %d elements \n"
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
	subq	$96, %rsp
	movl	%edi, -68(%rbp)
	movq	%rsi, -80(%rbp)
	movq	%rdx, -88(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_VfJR_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_VfJR_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_VfJR_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 109 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-VfJR--0
# 0 "" 2
#NO_APP
	movl	-68(%rbp), %eax
	movl	%eax, _TIG_IZ_VfJR_argc(%rip)
	movq	-80(%rbp), %rax
	movq	%rax, _TIG_IZ_VfJR_argv(%rip)
	movq	-88(%rbp), %rax
	movq	%rax, _TIG_IZ_VfJR_envp(%rip)
	nop
	movq	$0, -24(%rbp)
.L42:
	cmpq	$36, -24(%rbp)
	ja	.L44
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
	.long	.L26-.L8
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L22-.L8
	.long	.L44-.L8
	.long	.L44-.L8
	.long	.L44-.L8
	.long	.L21-.L8
	.long	.L44-.L8
	.long	.L20-.L8
	.long	.L19-.L8
	.long	.L44-.L8
	.long	.L44-.L8
	.long	.L18-.L8
	.long	.L44-.L8
	.long	.L44-.L8
	.long	.L44-.L8
	.long	.L44-.L8
	.long	.L17-.L8
	.long	.L44-.L8
	.long	.L16-.L8
	.long	.L44-.L8
	.long	.L15-.L8
	.long	.L44-.L8
	.long	.L14-.L8
	.long	.L44-.L8
	.long	.L44-.L8
	.long	.L13-.L8
	.long	.L44-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L9-.L8
	.long	.L44-.L8
	.long	.L44-.L8
	.long	.L7-.L8
	.text
.L14:
	movl	-36(%rbp), %eax
	cltq
	salq	$2, %rax
	leaq	-4(%rax), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movl	-36(%rbp), %edx
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	movq	-32(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	subl	$1, -36(%rbp)
	movq	$30, -24(%rbp)
	jmp	.L27
.L22:
	movl	-52(%rbp), %eax
	cmpl	%eax, -36(%rbp)
	jge	.L28
	movq	$1, -24(%rbp)
	jmp	.L27
.L28:
	movq	$36, -24(%rbp)
	jmp	.L27
.L12:
	movl	-40(%rbp), %eax
	cmpl	%eax, -36(%rbp)
	jle	.L30
	movq	$25, -24(%rbp)
	jmp	.L27
.L30:
	movq	$2, -24(%rbp)
	jmp	.L27
.L18:
	movl	-52(%rbp), %eax
	subl	$1, %eax
	movl	%eax, %esi
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
	addl	%eax, -36(%rbp)
	movq	$11, -24(%rbp)
	jmp	.L27
.L11:
	movl	-52(%rbp), %eax
	subl	$1, %eax
	movl	%eax, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-40(%rbp), %rdx
	leaq	-48(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-52(%rbp), %eax
	addl	$1, %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	realloc@PLT
	movl	-52(%rbp), %eax
	addl	$1, %eax
	movl	%eax, -52(%rbp)
	movl	-52(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -36(%rbp)
	movq	$30, -24(%rbp)
	jmp	.L27
.L21:
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	addl	$1, -36(%rbp)
	movq	$19, -24(%rbp)
	jmp	.L27
.L25:
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	addl	$1, -36(%rbp)
	movq	$4, -24(%rbp)
	jmp	.L27
.L15:
	movl	-44(%rbp), %eax
	cmpl	$4, %eax
	je	.L32
	cmpl	$4, %eax
	jg	.L33
	cmpl	$3, %eax
	je	.L34
	cmpl	$3, %eax
	jg	.L33
	cmpl	$1, %eax
	je	.L35
	cmpl	$2, %eax
	je	.L36
	jmp	.L33
.L32:
	movq	$3, -24(%rbp)
	jmp	.L37
.L34:
	movq	$32, -24(%rbp)
	jmp	.L37
.L36:
	movq	$14, -24(%rbp)
	jmp	.L37
.L35:
	movq	$31, -24(%rbp)
	jmp	.L37
.L33:
	movq	$21, -24(%rbp)
	nop
.L37:
	jmp	.L27
.L23:
	movl	$0, %edi
	call	exit@PLT
.L16:
	movq	$36, -24(%rbp)
	jmp	.L27
.L7:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-44(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$23, -24(%rbp)
	jmp	.L27
.L19:
	movl	-52(%rbp), %eax
	cmpl	%eax, -36(%rbp)
	jge	.L38
	movq	$28, -24(%rbp)
	jmp	.L27
.L38:
	movq	$10, -24(%rbp)
	jmp	.L27
.L17:
	movl	-52(%rbp), %eax
	cmpl	%eax, -36(%rbp)
	jge	.L40
	movq	$8, -24(%rbp)
	jmp	.L27
.L40:
	movq	$36, -24(%rbp)
	jmp	.L27
.L10:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -36(%rbp)
	movq	$19, -24(%rbp)
	jmp	.L27
.L13:
	movl	-36(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rdx, %rax
	movl	-36(%rbp), %edx
	movslq	%edx, %rdx
	salq	$2, %rdx
	leaq	-4(%rdx), %rcx
	movq	-32(%rbp), %rdx
	addq	%rcx, %rdx
	movl	(%rax), %eax
	movl	%eax, (%rdx)
	addl	$1, -36(%rbp)
	movq	$11, -24(%rbp)
	jmp	.L27
.L9:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-52(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-52(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -32(%rbp)
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	-52(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, -36(%rbp)
	movq	$4, -24(%rbp)
	jmp	.L27
.L20:
	movl	-52(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -52(%rbp)
	movq	$36, -24(%rbp)
	jmp	.L27
.L26:
	movq	$33, -24(%rbp)
	jmp	.L27
.L24:
	movl	-40(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-32(%rbp), %rax
	addq	%rax, %rdx
	movl	-48(%rbp), %eax
	movl	%eax, (%rdx)
	movq	$36, -24(%rbp)
	jmp	.L27
.L44:
	nop
.L27:
	jmp	.L42
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
