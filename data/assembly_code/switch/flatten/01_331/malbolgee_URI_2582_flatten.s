	.file	"malbolgee_URI_2582_flatten.c"
	.text
	.globl	_TIG_IZ_Z6wR_argv
	.bss
	.align 8
	.type	_TIG_IZ_Z6wR_argv, @object
	.size	_TIG_IZ_Z6wR_argv, 8
_TIG_IZ_Z6wR_argv:
	.zero	8
	.globl	_TIG_IZ_Z6wR_argc
	.align 4
	.type	_TIG_IZ_Z6wR_argc, @object
	.size	_TIG_IZ_Z6wR_argc, 4
_TIG_IZ_Z6wR_argc:
	.zero	4
	.globl	_TIG_IZ_Z6wR_envp
	.align 8
	.type	_TIG_IZ_Z6wR_envp, @object
	.size	_TIG_IZ_Z6wR_envp, 8
_TIG_IZ_Z6wR_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"DNSUEY!"
.LC1:
	.string	"HOST!"
.LC2:
	.string	"CRIPTONIZE"
.LC3:
	.string	"PROXYCITY"
.LC4:
	.string	"WIFI ANTENNAS"
.LC5:
	.string	"SALT"
.LC6:
	.string	"SERVERS"
.LC7:
	.string	"P.Y.N.G."
.LC8:
	.string	"%hu"
.LC9:
	.string	"%hu %hu"
.LC10:
	.string	"RAR?"
.LC11:
	.string	"ANSWER!"
.LC12:
	.string	"OFFLINE DAY"
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
	movq	$0, _TIG_IZ_Z6wR_envp(%rip)
	nop
.L2:
	movq	$0, _TIG_IZ_Z6wR_argv(%rip)
	nop
.L3:
	movl	$0, _TIG_IZ_Z6wR_argc(%rip)
	nop
	nop
.L4:
.L5:
#APP
# 125 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Z6wR--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_Z6wR_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_Z6wR_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_Z6wR_envp(%rip)
	nop
	movq	$32, -16(%rbp)
.L44:
	cmpq	$32, -16(%rbp)
	ja	.L47
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
	.long	.L25-.L8
	.long	.L24-.L8
	.long	.L23-.L8
	.long	.L47-.L8
	.long	.L47-.L8
	.long	.L22-.L8
	.long	.L47-.L8
	.long	.L47-.L8
	.long	.L47-.L8
	.long	.L47-.L8
	.long	.L21-.L8
	.long	.L47-.L8
	.long	.L47-.L8
	.long	.L20-.L8
	.long	.L47-.L8
	.long	.L47-.L8
	.long	.L19-.L8
	.long	.L18-.L8
	.long	.L17-.L8
	.long	.L48-.L8
	.long	.L47-.L8
	.long	.L15-.L8
	.long	.L47-.L8
	.long	.L14-.L8
	.long	.L13-.L8
	.long	.L12-.L8
	.long	.L11-.L8
	.long	.L10-.L8
	.long	.L47-.L8
	.long	.L47-.L8
	.long	.L9-.L8
	.long	.L47-.L8
	.long	.L7-.L8
	.text
.L17:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -16(%rbp)
	jmp	.L26
.L12:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -16(%rbp)
	jmp	.L26
.L9:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -16(%rbp)
	jmp	.L26
.L24:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -16(%rbp)
	jmp	.L26
.L14:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -16(%rbp)
	jmp	.L26
.L19:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -16(%rbp)
	jmp	.L26
.L13:
	movzwl	-20(%rbp), %eax
	cmpl	$10, %eax
	ja	.L27
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L29(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L29(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L29:
	.long	.L39-.L29
	.long	.L38-.L29
	.long	.L37-.L29
	.long	.L36-.L29
	.long	.L35-.L29
	.long	.L34-.L29
	.long	.L33-.L29
	.long	.L32-.L29
	.long	.L31-.L29
	.long	.L30-.L29
	.long	.L28-.L29
	.text
.L28:
	movq	$23, -16(%rbp)
	jmp	.L40
.L30:
	movq	$5, -16(%rbp)
	jmp	.L40
.L31:
	movq	$10, -16(%rbp)
	jmp	.L40
.L32:
	movq	$16, -16(%rbp)
	jmp	.L40
.L33:
	movq	$2, -16(%rbp)
	jmp	.L40
.L34:
	movq	$30, -16(%rbp)
	jmp	.L40
.L35:
	movq	$25, -16(%rbp)
	jmp	.L40
.L36:
	movq	$26, -16(%rbp)
	jmp	.L40
.L37:
	movq	$18, -16(%rbp)
	jmp	.L40
.L38:
	movq	$13, -16(%rbp)
	jmp	.L40
.L39:
	movq	$1, -16(%rbp)
	jmp	.L40
.L27:
	movq	$21, -16(%rbp)
	nop
.L40:
	jmp	.L26
.L15:
	movq	$0, -16(%rbp)
	jmp	.L26
.L11:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -16(%rbp)
	jmp	.L26
.L20:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -16(%rbp)
	jmp	.L26
.L7:
	leaq	-26(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$0, -16(%rbp)
	jmp	.L26
.L18:
	leaq	-22(%rbp), %rdx
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movzwl	-24(%rbp), %edx
	movzwl	-22(%rbp), %eax
	addl	%edx, %eax
	movw	%ax, -20(%rbp)
	movq	$24, -16(%rbp)
	jmp	.L26
.L10:
	cmpw	$0, -18(%rbp)
	je	.L42
	movq	$17, -16(%rbp)
	jmp	.L26
.L42:
	movq	$19, -16(%rbp)
	jmp	.L26
.L22:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -16(%rbp)
	jmp	.L26
.L21:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -16(%rbp)
	jmp	.L26
.L25:
	movzwl	-26(%rbp), %eax
	movw	%ax, -18(%rbp)
	movzwl	-26(%rbp), %eax
	subl	$1, %eax
	movw	%ax, -26(%rbp)
	movq	$27, -16(%rbp)
	jmp	.L26
.L23:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -16(%rbp)
	jmp	.L26
.L47:
	nop
.L26:
	jmp	.L44
.L48:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L46
	call	__stack_chk_fail@PLT
.L46:
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
